import time
import os
import sys
import pathlib
import uuid
import logging
import numpy as np
import torch
import random

from torch.utils.tensorboard import SummaryWriter

__all__ = ['generate_experiment_dir', 'dump_config_details_to_tensorboard',
           'dump_best_model_metrics_to_tensorboard', 'configure_logger', 'get_tensorboard_writer']


def generate_experiment_dir(model_dir, config, prefix_str=''):
    """
    Helper function to create checkpoint folder. We save
    model checkpoints using the provided model directory
    but we add a subfolder with format:
        Y-m-d_H:M_prefixStr_lr_batchSize_modelName_augmentation_numEpochs__UUID

    :param model_dir: Top directory of checkpoints
    :param prefix_str: Unique string for experiment
    :param config: Dictionary with config
    """
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    experiment_string = '{}_{}_{}_k{}_C{}_lr{}_Seed{}_{}_{}' \
        .format(timestamp, prefix_str+str(config['hilbert_level']),
                config['architecture'], config['kernel_size'], config['channels'], config['lr'], config['random_seed'],
                'augment' if config['augment'] else 'no-augment', config['max_epochs'], uuid.uuid4())

    # experiment_dir = os.path.join(os.path.curdir, experiment_string)
    experiment_dir = os.path.join(model_dir, experiment_string)
    pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    return experiment_dir


def dump_config_details_to_tensorboard(writer, config):
    """
    Helper function to dump initial configuration
    details of our model/training to Tensorboard.
    The function dumps all scalars int he config
    and omits all keys in `do_not_dump_in_tensorboard`

    :param writer: Tensorboard SummaryWritter object
    :param config: Dictionary with config
    """
    for k, v in config.items():
        if k not in config['do_not_dump_in_tensorboard']:
            writer.add_scalar('config/{}'.format(k), v, 0)


def dump_best_model_metrics_to_tensorboard(writer, phases, best_state):
    """
    Helper function to dump best metrics (loss and accuracy) of
    model, available in best_state

    :param writer: Tensorboard SummaryWritter object
    :param phases: `train` or `test`
    :param best_state: Dictionary with best state metrics
    """
    for phase in phases:
        writer.add_scalar('best_state/{}_loss'.format(phase), best_state['{}_loss'.format(phase)], 0)
        writer.add_scalar('best_state/{}_acc'.format(phase), best_state['{}_acc'.format(phase)], 0)
        # check if we have mIoU
        iou_key = '{}_mIoU'.format(phase)
        if iou_key in best_state:
            writer.add_scalar('best_state/{}_mIoU'.format(phase), best_state[iou_key], 0)
        # check if we have class accuracy
        class_acc_key = '{}_class_acc'.format(phase)
        if class_acc_key in best_state:
            writer.add_scalar('best_state/{}_acc_class'.format(phase), best_state[class_acc_key], 0)
    writer.add_scalar('best_state/convergence_epoch', best_state['convergence_epoch'], 0)


def configure_logger(model_dir, loglevel):
    """
    Configure logger on given level. Logging will occur on standard
    output and in a log file saved in model_dir.

    :param model_dir: Path to directory for saving logs
    :param loglevel: Logging level
    """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))

        # configure logger to display and save log data
    # log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    file_handler = logging.FileHandler(os.path.join(model_dir, '{}.log'.format(os.path.basename(model_dir))))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logging.root = logger


def get_tensorboard_writer(log_dir):
    """
    Create a Tensorboard writer object.

    :param log_dir: Checkpoints directory
    :return: SummaryWriter object
    """
    writer = SummaryWriter(log_dir=log_dir)

    return writer


def convert_label_to_one_hot(labels, num_categories):
    """
    Convert digit labels to one hot encoding

    :param labels: Px1 array of labels
    :param num_categories: Total number of categories available

    :return: One hot encoded label of size Pxnum_categories
    """
    label_one_hot = np.zeros((labels.shape[0], num_categories))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1

    return label_one_hot


# Tools to process raw S3DIS dataset (code taken from https://github.com/WangYueFt/dgcnn)

def sample_data(data, num_sample):
    """ data is in N x ...
      we want to keep num_samplexC of them.
      if N > num_sample, we will randomly keep num_sample of them.
      if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if N == num_sample:
        return data, range(N)
    elif N > num_sample:
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), np.concatenate((np.arange(N), sample))


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]

    return new_data, new_label


def room2blocks(data, label, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    """ Prepare block training data.
    Args:
      data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
        assumes the data is shifted (min point is origin) and aligned
        (aligned with XYZ axis)
      label: N size uint8 numpy array from 0-12
      num_point: int, how many points to sample in each block
      block_size: float, physical size of the block in meters
      stride: float, stride for block sweeping
      random_sample: bool, if True, we will randomly sample blocks in the room
      sample_num: int, if random sample, how many blocks to sample
        [default: room area]
      sample_aug: if random sample, how much aug
    Returns:
      block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
      block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
            continue

        block_data = data[cond, :]
        block_label = label[cond]

        # randomly subsample data
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)
        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
      for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:, 0:6]
    data[:, 3:6] /= 255.0
    label = data_label[:, -1].astype(np.uint8)
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    data_batch, label_batch = room2blocks(data, label, num_point, block_size, stride,
                                          random_sample, sample_num, sample_aug)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx + block_size / 2)
        data_batch[b, :, 1] -= (miny + block_size / 2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch


def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                   random_sample=False, sample_num=None, sample_aug=1):
    data_label = np.load(data_label_filename)

    return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                       random_sample, sample_num, sample_aug)


# Augmentation functions:
def augment_point_cloud(data):
    # rotate around z-axis (vertical axis)
    theta = np.random.uniform(low=0.0, high=2 * np.pi)
    cos = np.cos(theta)
    sin = np.sin(theta)
    ones = np.ones_like(cos)
    zeros = np.zeros_like(cos)
    r = np.stack([cos, zeros, sin, zeros, ones, zeros, -sin, zeros, cos], axis=0)
    r = np.reshape(r, (3, 3))
    rot_data = np.matmul(data[:, :3], r)

    # scale
    scale_min = 0.8
    scale_max = 1.2
    scale = np.random.uniform(low=scale_min, high=scale_max)
    scaled_data = rot_data * scale

    # add random noise
    augment_noise = 0.001
    noise = augment_noise * np.random.randn(rot_data.shape[0], rot_data.shape[1])
    jitter_data = scaled_data + noise

    # return new points with their available features
    out_data = np.concatenate((jitter_data, data[:, 3:]), axis=-1)

    return out_data


def augment_batch_point_cloud(data):
    num_batches = data.shape[0]
    # rotate around z-axis (vertical axis)
    theta = np.random.uniform(low=0.0, high=2 * np.pi)
    cos = np.cos(theta)
    sin = np.sin(theta)
    ones = np.ones_like(cos)
    zeros = np.zeros_like(cos)
    r = np.stack([cos, zeros, sin, zeros, ones, zeros, -sin, zeros, cos], axis=0)
    r = np.reshape(r, (3, 3))
    rot_data = np.matmul(data[:, :3], r)

    # scale
    scale_min = 0.8
    scale_max = 1.2
    scale = np.random.uniform(low=scale_min, high=scale_max)
    scaled_data = rot_data * scale

    # add random noise
    augment_noise = 0.001
    noise = augment_noise * np.random.randn(rot_data.shape[0], rot_data.shape[1])
    jitter_data = scaled_data + noise

    # return new points with their available features
    out_data = np.concatenate((jitter_data, data[:, 3:]), axis=-1)

    return out_data


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# get the number of trained parameters in a model

def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

