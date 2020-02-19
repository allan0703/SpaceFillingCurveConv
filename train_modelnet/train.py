from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import torch
import os

import torch.nn as nn

from tqdm import tqdm

from model.deeplab import deeplab
from model.unet import unet
from dataset_modelnet import ModelNet
import sys
sys.path.append('../')
import utils as utl
import metrics as metrics


def train(dataset, model_dir, writer):
    dataloaders = dataset.get_dataloaders()

    # we now set GPU training parameters
    # if the given index is not available then we use index 0
    # also when using multi gpu we should specify index 0
    if dataset.config.gpu_index + 1 > torch.cuda.device_count() or dataset.config.multi_gpu:
        dataset.config.gpu_index = 0

    logging.info('Using GPU cuda:{}, script PID {}'.format(dataset.config.gpu_index, os.getpid()))
    if dataset.config.multi_gpu:
        logging.info('Training on multi-GPU mode with {} devices'.format(torch.cuda.device_count()))
    device = torch.device('cuda:{}'.format(dataset.config.gpu_index))

    if dataset.config.model == 'unet':
        model = unet(input_size=dataset.config.num_feats, num_classes=dataset.config.num_classes,
                     kernel_size=dataset.config.kernel_size).to(device)
    else:
        model = deeplab(backbone=dataset.config.backbone, input_size=dataset.config.num_feats,
                        num_classes=dataset.config.num_classes, kernel_size=dataset.config.kernel_size,
                        sigma=dataset.config.sigma).to(device)

    # if use multi_gou then convert the model to DataParallel
    if dataset.config.multi_gpu:
        model = nn.DataParallel(model)

    # create optimizer, loss function, and lr scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=dataset.config.lr*100,
                                momentum=dataset.config.momentum, weight_decay=dataset.config.lr_decay)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, dataset.config.max_epochs, eta_min=dataset.config.lr)

    logging.info('Config {}'.format(dataset.config))
    logging.info('TB logs and checkpoint will be saved in {}'.format(model_dir))

    phases = ['train', 'test']
    # create metric trackers: we track lass, class accuracy, and overall accuracy
    trackers = {x: {'loss': metrics.LossMean(),
                    'acc': metrics.Accuracy(),
                    'cm': metrics.ConfusionMatrix(num_classes=int(dataset.config.num_classes))}
                for x in phases}

    # create initial best state object
    best_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'train_loss': float('inf'),
        'test_loss': float('inf'),
        'train_acc': 0.0,
        'test_acc': 0.0,
        'train_mean_class_acc': 0.0,
        'test_mean_class_acc': 0.0,
        'convergence_epoch': 0,
        'num_epochs_since_best_acc': 0
    }

    # now we train!
    for epoch in range(dataset.config.max_epochs):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # reset metrics
            trackers[phase]['loss'].reset()
            trackers[phase]['cm'].reset()

            for step_number, inputs in enumerate(tqdm(dataloaders[phase],
                                                      desc='[{}/{}] {} '.format(epoch + 1, dataset.config.max_epochs,
                                                                                phase))):
                data = inputs[0].to(device, dtype=torch.float).permute(0, 2, 1)
                coords = inputs[1].to(device, dtype=torch.float).permute(0, 2, 1)
                label = inputs[2].to(device, dtype=torch.long).squeeze(-1)
                edge_index = inputs[3].to(device, dtype=torch.long).permute(1,0,2,3)

                # compute gradients on train only
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(data, coords, edge_index)
                    loss = criterion(out, label)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # now we update metrics
                trackers[phase]['loss'].update(average_loss=loss, batch_size=data.size(0))
                trackers[phase]['cm'].update(y_true=label, y_logits=out)

            logging.info('Computing accuracy...')

            # compare with my metrics
            epoch_loss = trackers[phase]['loss'].result()
            epoch_overall_acc = trackers[phase]['cm'].result(metric='accuracy')
            epoch_mean_class_acc = trackers[phase]['cm'].result(metric='class_accuracy').mean()

            logging.info('--------------------------------------------------------------------------------')
            logging.info('[{}/{}] {} Loss: {:.2e}. Overall Acc: {:.4f}. Mean Class Acc {:.4f}'
                         .format(epoch + 1, dataset.config.max_epochs, phase, epoch_loss,
                                 epoch_overall_acc, epoch_mean_class_acc))

            logging.info('--------------------------------------------------------------------------------')

            # we update our learning rate scheduler if loss does not improve
            if phase == 'test' and scheduler:
                scheduler.step(epoch_loss)
                writer.add_scalar('params/lr', optimizer.param_groups[0]['lr'], epoch + 1)

            writer.add_scalar('loss/epoch_{}'.format(phase), epoch_loss, epoch + 1)
            writer.add_scalar('mean_class_acc/epoch_{}'.format(phase), epoch_mean_class_acc, epoch + 1)
            writer.add_scalar('acc_all/epoch_{}'.format(phase), epoch_overall_acc, epoch + 1)

        # after each epoch we update best state values as needed
        # first we save our state when we get better test accuracy
        if best_state['test_acc'] > trackers['test']['cm'].result(metric='accuracy'):
            best_state['num_epochs_since_best_acc'] += 1
        else:
            logging.info('Got a new best model with overall acc {:.4f}'
                         .format(trackers['test']['cm'].result(metric='accuracy')))
            best_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'train_loss': trackers['train']['loss'].result(),
                'test_loss': trackers['test']['loss'].result(),
                'train_acc': trackers['train']['cm'].result(metric='accuracy'),
                'test_acc': trackers['test']['cm'].result(metric='accuracy'),
                'train_mean_class_acc': trackers['train']['cm'].result(metric='class_accuracy').mean(),
                'test_mean_class_acc': trackers['test']['cm'].result(metric='class_accuracy').mean(),
                'convergence_epoch': epoch + 1,
                'num_epochs_since_best_acc': 0
            }

            file_name = os.path.join(model_dir, 'best_state.pth')
            torch.save(best_state, file_name)
            logging.info('saved checkpoint in {}'.format(file_name))

        # we check for early stopping when we have trained a min number of epochs
        if epoch >= dataset.config.min_epochs and best_state['num_epochs_since_best_acc'] >= dataset.config.early_stopping:
            logging.info('Accuracy did not improve for {} iterations!'.format(dataset.config.early_stopping))
            logging.info('[Early stopping]')
            break

    utl.dump_best_model_metrics_to_tensorboard(writer, phases, best_state)

    logging.info('************************** DONE **************************')


def main(args):
    # given program arguments, generate a config file
    dataset = ModelNet(args)
    # create a checkpoint directory
    model_dir = dataset.experiment_dir

    # configure logger
    utl.configure_logger(model_dir, args.loglevel.upper())

    # get Tensorboard writer object
    writer = utl.get_tensorboard_writer(log_dir=model_dir)

    dataset.config.dump_to_tensorboard(writer=writer)

    print(dataset.config)
    utl.set_seed(dataset.config.seed)

    # configurations = []
    # for i in dir(dataset.config):
    #     if i[0] !='_':
    #         configurations.append(dataset.config)

    # logging.info(dataset.config)
    train(dataset=dataset, model_dir=model_dir, writer=writer)

    # close Tensorboard writer
    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a 1D segmentation network (Deeplab v3) to perform part segmentation '
                                        'on Shapenet pointclouds.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root_dir', required=True, type=str,
                        help='root directory containing S3DIS data')
    parser.add_argument('--model_dir', default='../log', type=str,
                        help='root directory containing S3DIS data')
    parser.add_argument('--test_area', default=5, type=int,
                        help='area to use for testing')
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help='use multiple GPUs (all available)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='index of GPU to use (0-indexed); if multi_gpu then value is ignored')
    parser.add_argument('--model', default=None, type=str,
                        help='either deeplab or unet')
    parser.add_argument('--backbone', default=None, type=str,
                        help='backbone to use for deeplab (xception, resnet101, resnet18')
    parser.add_argument('--kernel_size', default=None, type=int,
                        help='odd value for kernel size')
    parser.add_argument('--num_feats', default=None, type=int,
                        help='number of input features to use (4, 5, or 9')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='batch size for training')
    parser.add_argument('--lr', default=None, type=float,
                        help='learning rate')
    parser.add_argument('--dilation', default=1, type=int,
                        help='dilation to use')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='whether to augment training data')
    parser.add_argument('--bias', default=False, action='store_true',
                        help='use bias in convolutions')
    parser.add_argument('--random_seed', default=None, type=int,
                        help='optional random seed')
    parser.add_argument('--loglevel', default='INFO', type=str,
                        help='logging level')

    args = parser.parse_args()

    main(args)
