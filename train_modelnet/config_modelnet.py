import numpy as np


def generate_config(args):
    """
    Function to create a configuration file used in training
    Z-Conv part segmentation network. Random parameters are
    created for hyperparameter search.

    :param args: main function arguments used for storing information
    :return: configuration dictionary used for training
    """

    kernel_size = args.kernel_size
    lr = args.lr
    random_seed = args.random_seed
    batch_size = args.batch_size
    sigma = args.sigma

    if args.hs:
        kernel_size = np.random.choice([3, 9, 15, 21, 27])
        lr = np.random.choice([1e-3, 1e-4])
        random_seed = random_seed * np.random.randint(1000)
        batch_size = int(np.random.choice([8, 16]))
        sigma = np.random.choice([0.02, 0.05, 0.1, 0.5, 1.5, 2.5])

    config = {
        'lr': lr,
        'batch_size': int(batch_size),
        'kernel_size': int(kernel_size),
        'in_channels': 3,
        'sigma': sigma,
        'augment': args.augment,
        'n_points': 2048,
        'bias': args.bias,
        'num_classes': 0,
        'max_epochs': 300,
        'min_epochs': 100,
        'lr_decay': 0.9,
        'lr_patience': 1,
        'early_stopping': 20,
        'gpu_index': args.gpu,
        'multi_gpu': args.multi_gpu,
        'root_dir': args.root_dir,
        'model_dir': args.model_dir,
        'architecture': 'resnet101',
        'random_seed': random_seed,
        'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'model', 'order',
                                       'category', 'dataset', 'data_loading_function',
                                       'backbone', 'root_dir', 'model_dir', 'architecture'],
    }

    return config


