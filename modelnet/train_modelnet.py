from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import torch
import os

import numpy as np
import torch.nn as nn

from tqdm import tqdm

import config as cfg
from modelnet import dataset_modelnet as ds, resnet as res

import utils as utl
import metrics as metrics

# https://gist.github.com/ModarTensai/2328b13bdb11c6309ba449195a6b551a
# np.random.seed(0)
# random.seed(0)

# https://pytorch.org/docs/stable/notes/randomness.html
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# non-determinism is unavoidable in some functions that use atomicAdd for example
# such as torch.nn.functional.embedding_bag() and torch.bincount()


def train(config, model_dir, writer):
    """
    Function train and evaluate a part segmentation model for
    the Shapenet dataset. The training parameters are specified
    in the config file (for more details see config/config.py).

    :param config: Dictionary with configuration paramters
    :param model_dir: Checkpoint save directory
    :param writer: Tensorboard SummaryWritter object
    """
    phases = ['train', 'test']

    datasets, dataloaders = ds.get_modelnet40_dataloaders(root_dir=args.root_dir,
                                                          phases=phases,
                                                          batch_size=config['batch_size'],
                                                          augment=config['augment'])

    # add number of classes to config
    config['num_classes'] = 40

    # we now set GPU training parameters
    # if the given index is not available then we use index 0
    # also when using multi gpu we should specify index 0
    if config['gpu_index'] + 1 > torch.cuda.device_count() or config['multi_gpu']:
        config['gpu_index'] = 0

    logging.info('Using GPU cuda:{}, script PID {}'.format(config['gpu_index'], os.getpid()))
    if config['multi_gpu']:
        logging.info('Training on multi-GPU mode with {} devices'.format(torch.cuda.device_count()))
    device = torch.device('cuda:{}'.format(config['gpu_index']))

    # we load the model defined in the config file
    model = res.resnet101(in_channels=config['in_channels'], num_classes=config['num_classes'],
                          kernel_size=config['kernel_size']).to(device)

    # if use multi_gpu then convert the model to DataParallel
    if config['multi_gpu']:
        model = nn.DataParallel(model)

    # create optimizer, loss function, and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=config['lr_decay'],
                                                           patience=config['lr_patience'],
                                                           verbose=True)

    logging.info('Config {}'.format(config))
    logging.info('TB logs and checkpoint will be saved in {}'.format(model_dir))

    utl.dump_config_details_to_tensorboard(writer, config)

    # create metric trackers: we track lass, class accuracy, and overall accuracy
    trackers = {x: {'loss': metrics.LossMean(),
                    'acc': metrics.Accuracy(),
                    'iou': None,
                    'cm': metrics.ConfusionMatrix(num_classes=int(config['num_classes']))}
                for x in phases}

    # create initial best state object
    best_state = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'train_loss': float('inf'),
        'test_loss': float('inf'),
        'train_acc': 0.0,
        'test_acc': 0.0,
        'train_class_acc': 0.0,
        'test_class_acc': 0.0,
        'convergence_epoch': 0,
        'num_epochs_since_best_acc': 0
    }

    # now we train!
    for epoch in range(config['max_epochs']):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # reset metrics
            trackers[phase]['loss'].reset()
            trackers[phase]['cm'].reset()

            for step_number, (data, label) in enumerate(tqdm(dataloaders[phase],
                                                        desc='[{}/{}] {} '.format(epoch + 1, config['max_epochs'],
                                                                                  phase))):
                data = data.to(device, dtype=torch.float).permute(0, 2, 1)
                label = label.to(device, dtype=torch.long).squeeze()

                # compute gradients on train only
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(data)
                    loss = criterion(out, label)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # now we update metrics
                trackers[phase]['loss'].update(average_loss=loss, batch_size=data.size(0))
                trackers[phase]['cm'].update(y_true=label, y_logits=out)

            # logging.info('Computing accuracy...')

            # compare with my metrics
            epoch_loss = trackers[phase]['loss'].result()
            epoch_overall_acc = trackers[phase]['cm'].result(metric='accuracy')
            epoch_class_acc = trackers[phase]['cm'].result(metric='class_accuracy').mean()

            # we update our learning rate scheduler if loss does not improve
            if phase == 'test' and scheduler:
                scheduler.step(epoch_loss)
                writer.add_scalar('params/lr', optimizer.param_groups[0]['lr'], epoch + 1)

            # log current results and dump in Tensorboard
            logging.info('[{}/{}] {} Loss: {:.2e}. Overall Acc: {:.4f}. Class Acc {:.4f}'
                         .format(epoch + 1, config['max_epochs'], phase, epoch_loss,
                                 epoch_overall_acc, epoch_class_acc))

            writer.add_scalar('loss/epoch_{}'.format(phase), epoch_loss, epoch + 1)
            writer.add_scalar('acc_class/epoch_{}'.format(phase), epoch_class_acc, epoch + 1)
            writer.add_scalar('acc_all/epoch_{}'.format(phase), epoch_overall_acc, epoch + 1)

        # after each epoch we update best state values as needed
        # first we save our state when we get better test accuracy
        if best_state['test_acc'] > trackers['test']['cm'].result(metric='accuracy'):
            best_state['num_epochs_since_best_acc'] += 1
        else:
            logging.info('Got a new best model with accuracy {:.4f}'
                         .format(trackers['test']['cm'].result(metric='accuracy')))
            best_state = {
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'train_loss': trackers['train']['loss'].result(),
                'test_loss': trackers['test']['loss'].result(),
                'train_acc': trackers['train']['cm'].result(metric='accuracy'),
                'test_acc': trackers['test']['cm'].result(metric='accuracy'),
                'train_class_acc': trackers['train']['cm'].result(metric='class_accuracy').mean(),
                'test_class_acc': trackers['test']['cm'].result(metric='class_accuracy').mean(),
                'convergence_epoch': epoch + 1,
                'num_epochs_since_best_acc': 0
            }

            file_name = os.path.join(model_dir, 'best_state.pth')
            torch.save(best_state, file_name)
            logging.info('saved checkpoint in {}'.format(file_name))

        # we check for early stopping when we have trained a min number of epochs
        if epoch >= config['min_epochs'] and best_state['num_epochs_since_best_acc'] >= config['early_stopping']:
            logging.info('Accuracy did not improve for {} iterations!'.format(config['early_stopping']))
            logging.info('[Early stopping]')
            break

    utl.dump_best_model_metrics_to_tensorboard(writer, phases, best_state)

    logging.info('************************** DONE **************************')


def main(args):
    # given program arguments, generate a config file
    config = cfg.generate_config(args)

    config['in_channels'] = 3

    # if given a best state then we load it's config
    if args.state:
        logging.info('loading config from {}'.format(args.state))
        best_state = torch.load(args.state)
        config = best_state['config']

    # create a checkpoint directory
    model_dir = utl.generate_experiment_dir(args.model_dir, config, prefix_str='modelnet40-hilbert')

    # configure logger
    utl.configure_logger(model_dir, args.loglevel.upper())

    # get Tensorboard writer object
    writer = utl.get_tensorboard_writer(log_dir=model_dir)

    train(config=config, model_dir=model_dir, writer=writer)

    # close Tensorboard writer
    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a point cloud classification network using 1D convs and hilbert order.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root_dir', type=str, default='/data/sfc/modelnet40_ply_hdf5_2048',
                        help='root directory containing ModelNet40 data')
    parser.add_argument('--model_dir', type=str, default='log/',
                        help='root directory containing ModelNet40 data')
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help='use multiple GPUs (all available)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='index of GPU to use (0-indexed); if multi_gpu then value is ignored')
    parser.add_argument('--state', default=None, type=str,
                        help='path for best state to load')
    # parser.add_argument('--batch_size', default=None, type=int, help='batch size')
    # parser.add_argument('--kernel_size', default=None, type=int,
    #                     help='odd value for kernel size')
    # parser.add_argument('--lr', default=None, type=float,
    #                     help='learning rate')
    # parser.add_argument('--bias', default=None, action='store_true',
    #                     help='use bias in convolutions')
    # parser.add_argument('--augment', default=None, action='store_true',
    #                     help='use augmentation in training')
    # parser.add_argument('--random_seed', default=None, type=int,
    #                     help='optional random seed')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--kernel_size', default=15, type=int,
                        help='odd value for kernel size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--bias', default=True, action='store_true',
                        help='use bias in convolutions')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='use augmentation in training')
    parser.add_argument('--random_seed', default=1, type=int,
                        help='optional random seed')
    parser.add_argument('--loglevel', default='INFO', type=str,
                        help='logging level')

    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    main(args)
