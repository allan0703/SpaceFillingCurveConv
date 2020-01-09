import numpy as np


def generate_config(args):
    """
    Function to create a configuration file used in training
    Z-Conv part segmentation network. Random parameters are
    created for hyperparameter search.

    :param args: main function arguments used for storing information
    :return: configuration dictionary used for training
    """
    kernel_size = np.random.choice([3, 9, 15, 21, 27])
    if args.kernel_size is not None:
        kernel_size = args.kernel_size

    lr = np.random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    if args.lr is not None:
        lr = args.lr

    # bias = np.random.choice([True, False])
    bias = False
    if args.bias is not None:
        bias = args.bias

    batch_size = np.random.choice([8, 16, 32, 64, 128])
    if args.batch_size is not None:
        batch_size = args.batch_size

    augment = np.random.choice([True, False])
    if args.augment is not None:
        augment = args.augment

    config = {
        'lr': lr,
        'batch_size': int(batch_size),
        'kernel_size': int(kernel_size),
        'in_channels': args.in_channels,
        'augment': augment,
        'bias': bias,
        'num_classes': 0,
        'category': None,
        'max_epochs': 300,
        'min_epochs': 100,
        'lr_decay': 0.9,
        'lr_patience': 1,
        'early_stopping': 20,
        'gpu_index': args.gpu,
        'multi_gpu': args.multi_gpu,
        'root_dir': args.root_dir,
        'model_dir': args.model_dir,
        'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'model', 'order',
                                       'category', 'dataset', 'data_loading_function',
                                       'backbone', 'root_dir', 'model_dir'],
    }

    return config
