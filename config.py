import numpy as np


def generate_config(args):
    """
    Function to create a configuration file used in training
    Z-Conv part segmentation network. Random parameters are
    created for hyperparameter search.

    :param args: main function arguments used for storing information
    :return: configuration dictionary used for training
    """

    if args.hyperpara_search:
        kernel_size = np.random.choice([1, 3, 9, 15, 21, 27])
        # lr = np.random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        random_seed = np.random.randint(0, 1000, 1)
    else:
        kernel_size = args.kernel_size
        lr = args.lr
        random_seed = args.random_seed

    config = {
        'lr': lr,
        'batch_size': args.batch_size,
        'kernel_size': int(kernel_size),
        'in_channels': args.in_channels,
        'channels': args.channels,
        'augment': args.augment,
        'bias': args.bias,
        'num_classes': 0,
        'category': args.category,
        'max_epochs': 300,
        'min_epochs': 100,
        'lr_decay': 0.9,
        'lr_patience': 1,
        'early_stopping': 20,
        'gpu_index': args.gpu,
        'multi_gpu': args.multi_gpu,
        'root_dir': args.root_dir,
        'model_dir': args.model_dir,
        'hilbert_level': args.hilbert_level,
        'architecture': args.architecture,
        'random_seed': random_seed,
        'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'model', 'order',
                                       'category', 'dataset', 'data_loading_function',
                                       'backbone', 'root_dir', 'model_dir', 'architecture'],
    }

    return config


