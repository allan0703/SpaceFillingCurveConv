import numpy as np
import os


class S3DISConfig:
    # Configuration of dataset and model:
    dataset = 'S3DIS'
    model = 'deeplab'  # np.random.choice(['unet', 'deeplab'])
    backbone = 'resnet18'  # np.random.choice(['resnet18', 'resnet101'])  # np.random.choice(['xception', 'resnet18', 'resnet101'])
    num_classes = 13

    # Hyperparameters for training:
    test_area = 5  # np.random.choice([1, 2, 3, 4, 5, 6])
    kernel_size = np.random.choice([3, 5, 9, 15])
    num_feats = 4  # np.random.choice([4, 9])  # np.random.choice([4, 5, 9])
    lr = np.random.choice([1e-3, 1e-4])
    batch_size = int(np.random.choice([8, 16]))
    sigma = np.random.choice([0.02, 0.05, 0.1, 0.5, 1.5, 2.5])
    augment = True  # np.random.choice([True, False])
    bias = False  # np.random.choice([True, False])
    p = 7  # hilbert order

    # Training setup:
    max_epochs = 300
    min_epochs = 100
    lr_decay = 0.9
    lr_patience = 1
    early_stopping = 5

    # GPUs:
    gpu_index = 0
    multi_gpu = False

    # Paths to load data and save checkpoints:
    # These values must be set by the application
    root_dir = ''
    model_dir = ''

    def save(self, path):
        """
        Save all the parameters from the config in given path

        :param path: full path to save parameters
        """
        with open(os.path.join(path, 'config.txt'), 'w') as f:
            f.write('# -----------------------------------#\n')
            f.write('#      Configuration Parameters      #\n')
            f.write('# -----------------------------------#\n\n')

            # dataset and model
            f.write('# Dataset and Model\n')
            f.write('# -----------------------------------#\n\n')
            f.write('dataset = {:s}\n'.format(self.dataset))
            f.write('model = {:s}\n'.format(self.model))
            f.write('backbone = {:s}\n'.format(self.backbone))
            f.write('num_classes = {:d}\n\n'.format(self.num_classes))

            # hyperparameters for training
            f.write('# Training hyperparameters\n')
            f.write('# -----------------------------------#\n\n')
            f.write('test_area = {:d}\n'.format(self.test_area))
            f.write('kernel_size = {:d}\n'.format(self.kernel_size))
            f.write('num_feats = {:d}\n'.format(self.num_feats))
            f.write('lr = {:.2e}\n'.format(self.lr))
            f.write('batch_size = {:d}\n'.format(self.batch_size))
            f.write('sigma = {:f}\n'.format(self.sigma))
            f.write('augment = {}\n'.format(self.augment))
            f.write('bias = {}\n\n'.format(self.bias))
            f.write('p = {:d}\n\n'.format(self.p))

            # training setup
            f.write('# Training setup\n')
            f.write('# -----------------------------------#\n\n')
            f.write('min_epochs = {:d}\n'.format(self.min_epochs))
            f.write('max_epochs = {:d}\n'.format(self.max_epochs))
            f.write('lr_decay = {:f}\n'.format(self.lr_decay))
            f.write('lr_patience = {:f}\n'.format(self.lr_patience))
            f.write('early_stopping = {:d}\n'.format(self.early_stopping))

            # GPUs
            f.write('# GPUs\n')
            f.write('# -----------------------------------#\n\n')
            f.write('gpu_index = {:d}\n'.format(self.gpu_index))
            f.write('multi_gpu = {}\n\n'.format(self.multi_gpu))

            # Paths to data and checkpoints
            f.write('# Paths to data and checkpoints\n')
            f.write('# -----------------------------------#\n\n')
            f.write('root_dir = {:s}\n'.format(self.root_dir))
            f.write('model_dir = {:s}\n'.format(self.model_dir))

    def load(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Class variable dictionary
        for line in lines:
            line_info = line.split()
            # print(line_info)
            if len(line_info) > 1 and line_info[0] != '#':
                # if line_info[0] == 'model':
                #     self.model = [b for b in line_info[2:]]
                # elif line_info[0] == 'dataset':
                #     self.dataset = [b for b in line_info[2:]]
                # elif line_info[0] == 'backbone':
                #     self.backbone = [b for b in line_info[2:]]
                # elif line_info[0] == 'root_dir':
                #     self.root_dir = [b for b in line_info[2:]]
                # elif line_info[0] == 'model_dir':
                #     self.model_dir = [b for b in line_info[2:]]
                # else:
                attr_type = type(getattr(self, line_info[0]))
                if attr_type == bool:
                    val = True
                    if line_info[2] == 'False':
                        val = False
                    setattr(self, line_info[0], val)
                else:
                    setattr(self, line_info[0], attr_type(line_info[2]))

    def dump_to_tensorboard(self, writer):
        writer.add_scalar('config/test_area', self.test_area, 0)
        writer.add_scalar('config/kernel_size', self.kernel_size, 0)
        writer.add_scalar('config/num_feats', self.num_feats, 0)
        writer.add_scalar('config/lr', self.lr, 0)
        writer.add_scalar('config/batch_size', self.batch_size, 0)
        writer.add_scalar('config/sigma', self.sigma, 0)
        writer.add_scalar('config/augment', self.augment, 0)
        writer.add_scalar('config/bias', self.bias, 0)
        writer.add_scalar('config/p', self.p, 0)
        writer.add_scalar('config/min_epochs', self.min_epochs, 0)
        writer.add_scalar('config/max_epochs', self.max_epochs, 0)
        writer.add_scalar('config/lr_decay', self.lr_decay, 0)
        writer.add_scalar('config/lr_patience', self.lr_patience, 0)
        writer.add_scalar('config/early_stopping', self.early_stopping, 0)


if __name__ == '__main__':
    config = S3DISConfig()

    print(config)

    config.save('.')

