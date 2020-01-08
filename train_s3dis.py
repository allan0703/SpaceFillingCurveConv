from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import torch
import os

import numpy as np
import torch.nn as nn

from tqdm import tqdm

import config_modelnet as cfg
import dataset_s3dis as ds
import resnet_seg as res

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

    # todo: rewrite the loading in pytorch geometric way.
    datasets, dataloaders, num_classes = ds.get_s3dis_dataloaders(root_dir=args.root_dir,
                                                                  phases=phases,
                                                                  batch_size=config['batch_size'],
                                                                  category=config['augment'],
                                                                  augment=config['augment'])

    # add number of classes to config
    config['num_classes'] = num_classes

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
    # todo: now the code is IO bound. No matter which network we use, it is similar speed.
    model = res.resnet18(input_size=config['input_size'], num_classes=config['num_classes'],
                         kernel_size=config['kernel_size']).to(device)

    # if use multi_gpu then convert the model to DataParallel
    if config['multi_gpu']:
        model = nn.DataParallel(model)

    # create optimizer, loss function, and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=config['lr_decay'],
                                                           patience=config['lr_patience'],
                                                           verbose=True)

    logging.info('Config {}'.format(config))
    logging.info('TB logs and checkpoint will be saved in {}'.format(model_dir))

    utl.dump_config_details_to_tensorboard(writer, config)

    # create metric trackers: we track lass, class accuracy, and overall accuracy
    # todo: iou none?
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
                label = label.to(device, dtype=torch.long)

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
            epoch_iou = trackers[phase]['cm'].result(metric='iou').mean()

            # we update our learning rate scheduler if loss does not improve
            if phase == 'test' and scheduler:
                scheduler.step(epoch_loss)
                writer.add_scalar('params/lr', optimizer.param_groups[0]['lr'], epoch + 1)

            # log current results and dump in Tensorboard
            logging.info('[{}/{}] {} Loss: {:.2e}. mIOU {:.4f}'
                         .format(epoch + 1, config['max_epochs'], phase, epoch_loss, epoch_iou))

            writer.add_scalar('loss/epoch_{}'.format(phase), epoch_loss, epoch + 1)
            writer.add_scalar('mIoU/epoch_{}'.format(phase), epoch_iou, epoch + 1)

        # after each epoch we update best state values as needed
        # first we save our state when we get better test accuracy
        if best_state['test_acc'] > trackers['test']['cm'].result(metric='iou'):
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
                'train_acc': trackers['train']['cm'].result(metric='iou').mean(),
                'test_acc': trackers['test']['cm'].result(metric='iou').mean(),
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

    parser.add_argument('--root_dir', type=str, default='/data/sfc/indoor3d_sem_seg_hdf5_data',
                        help='root directory containing S3DIS data')
    parser.add_argument('--model_dir', type=str, default='log/')
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help='use multiple GPUs (all available)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='index of GPU to use (0-indexed); if multi_gpu then value is ignored')
    parser.add_argument('--state', default=None, type=str,
                        help='path for best state to load')
    parser.add_argument('--input_size', default=9, type=int, help='batch size')
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
    parser.add_argument('--loglevel', default='INFO', type=str, help='logging level')
    parser.add_argument('--category', default=5, type=int, help='Area used for test set (1, 2, 3, 4, 5, or 6)')

    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    main(args)
