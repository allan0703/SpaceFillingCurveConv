from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import torch
import torch.nn as nn

from tqdm import tqdm
import config as cfg
import load_partnet_preorder as ds
import architecture_knn as res

import utils as utl
import metrics as metrics


# https://gist.github.com/ModarTensai/2328b13bdb11c6309ba449195a6b551a
# np.random.seed(0)
# random.seed(0)

# https://pytorch.org/docs/stable/notes/randomness.html


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
    # phases = ['test', 'train']
    # todo: config, and check the hilbert curve
    datasets, dataloaders, num_classes = ds.get_s3dis_dataloaders(root_dir=config['root_dir'],
                                                                  phases=phases,
                                                                  batch_size=config['batch_size'],
                                                                  category=config['category'],
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
    model = res.sfc_resnet_8(in_channels=config['in_channels'], num_classes=config['num_classes'],
                             kernel_size=config['kernel_size'], channels=config['channels'],
                             use_tnet=config['use_tnet'], n_points=config['n_points']).to(device)
    logging.info('the number of params is {: .2f} M'.format(utl.count_model_params(model) / (1e6)))
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
                                                           verbose=True)  # verbose. recommended to use.

    logging.info('Config {}'.format(config))
    logging.info('TB logs and checkpoint will be saved in {}'.format(model_dir))

    utl.dump_config_details_to_tensorboard(writer, config)

    # create metric trackers: we track lass, class accuracy, and overall accuracy
    trackers = {x: {'loss': metrics.LossMean(),
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
        'train_mIoU': 0.0,
        'test_mIoU': 0.0,
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

            # use tqdm to show progress and print message
            # this is for loadding our new data format
            for step_number, batchdata in enumerate(tqdm(dataloaders[phase],
                                                         desc='[{}/{}] {} '.format(epoch + 1, config['max_epochs'],
                                                                                   phase))):
                data = batchdata.pos.transpose(1, 2).to(device, dtype=torch.float)
                label = batchdata.y.to(device, dtype=torch.long)
                # should we release the memory?
                # todo: add data augmentation

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

            # compare with my metrics
            epoch_loss = trackers[phase]['loss'].result()
            epoch_iou = trackers[phase]['cm'].result(metric='iou').mean()

            # we update our learning rate scheduler if loss does not improve
            if phase == 'train' and scheduler:
                scheduler.step(epoch_loss)
                writer.add_scalar('params/lr', optimizer.param_groups[0]['lr'], epoch + 1)

            # log current results and dump in Tensorboard
            logging.info('[{}/{}] {} Loss: {:.2e}. mIOU {:.4f} \t best testing mIOU {:.4f}'
                         .format(epoch + 1, config['max_epochs'], phase, epoch_loss, epoch_iou,
                                 best_state['test_mIoU']))

            writer.add_scalar('loss/epoch_{}'.format(phase), epoch_loss, epoch + 1)
            writer.add_scalar('mIoU/epoch_{}'.format(phase), epoch_iou, epoch + 1)

        # after each epoch we update best state values as needed
        # first we save our state when we get better test accuracy
        test_iou = trackers['test']['cm'].result(metric='iou').mean()
        if best_state['test_mIoU'] > test_iou:
            best_state['num_epochs_since_best_acc'] += 1
        else:
            logging.info('Got a new best model with iou {:.4f}'
                         .format(test_iou))
            best_state = {
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'train_loss': trackers['train']['loss'].result(),
                'test_loss': trackers['test']['loss'].result(),
                'train_mIoU': trackers['train']['cm'].result(metric='iou').mean(),
                'test_mIoU': test_iou,
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
    model_dir = utl.generate_experiment_dir(args.model_dir, config, prefix_str='S3DIS-hilbert')

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
    parser.add_argument('--root_dir', type=str, default='/data/sfc/partnet',
                        help='root directory containing Partnet data')
    parser.add_argument('--model_dir', type=str, default='log/')
    parser.add_argument('--multi_gpu', default=False, action='store_true', help='use multiple GPUs (all available)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='index of GPU to use (0-indexed); if multi_gpu then value is ignored')
    parser.add_argument('--state', default=None, type=str, help='path for best state to load (pre-trained model)')
    parser.add_argument('--in_channels', default=3, type=int, help='input channel size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument('--channels', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--bias', action='store_true', help='use bias in convolutions')
    parser.add_argument('--augment', action='store_true', help='use augmentation in training')
    parser.add_argument('--random_seed', default=1, type=int, help='optional random seed')
    parser.add_argument('--loglevel', default='INFO', type=str, help='logging level')
    parser.add_argument('--category', default='Bed', type=str, help='category for training a model')
    parser.add_argument('--hilbert_level', default=7, type=int, help='hilbert curve level')
    parser.add_argument('--architecture', default='res8-knn', type=str, help='architecture')
    parser.add_argument('--hyperpara_search', action='store_true', help='random choose a hyper parameter')
    parser.add_argument('--use_tnet', default=False, type=bool, help='random choose a hyper parameter')
    # parser.add_argument('--use_knn', default=False, type=bool, help='random choose a hyper parameter')
    parser.add_argument('--n_points', default=10000, type=int)
    args = parser.parse_args()

    main(args)
