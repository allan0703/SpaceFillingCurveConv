import numpy as np
import utils as utl


def generate_config(args):
    """
    Function to create a configuration file used in training
    Z-Conv part segmentation network. Random parameters are
    created for hyperparameter search.
    :param args: main function arguments used for storing information
    :return: configuration dictionary used for training
    """

    if args.hyperpara_search:
        #  kernel_size = np.random.choice([1, 3, 5, 9, 15])
        #  knn = np.random.choice([1, 3, 5, 9, 15])
        args.random_seed = np.random.randint(0, 1000, 1)[0]
        args.kernel_size = np.random.choice([3, 5, 9, 15])
        args.sigma = np.random.choice([0.02, 0.05, 0.1, 0.5, 1.5, 2.5])
        # args.augment = True  # np.random.choice([True, False])

    config = {
        'backbone': args.backbone,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'kernel_size': args.kernel_size,
        'in_channels': args.in_channels,
        'channels': args.channels,
        'augment': args.augment,
        'n_points': args.n_points,
        'bias': args.bias,
        'num_classes': 0,  # will change according to the dataset.
        'category': args.category,
        'level': args.level,
        'max_epochs': 200,
        'min_epochs': 200,
        'lr_decay': 0.8,
        'lr_patience': 10,
        'early_stopping': 40,
        'gpu_index': args.gpu,
        'multi_gpu': args.multi_gpu,
        'root_dir': args.root_dir,
        'model_dir': args.model_dir,
        'hilbert_level': args.hilbert_level,
        'use_tnet': args.use_tnet,
        'use_weighted_conv': args.use_weighted_conv,
        'sigma': args.sigma,
        'use_knn': args.use_knn,
        'knn': args.knn,
        'random_seed': args.random_seed,
        'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'model', 'order',
                                       'category', 'dataset', 'data_loading_function',
                                       'backbone', 'root_dir', 'model_dir', 'architecture'],
    }

    utl.set_seed(args.random_seed)
    return config



