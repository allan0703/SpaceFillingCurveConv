from argparse import ArgumentParser
import numpy as np
import os
from tqdm import tqdm, trange
import h5py
import logging
import time
import uuid
import pathlib
# from ..utils import vis_points
from torch.utils.data import Dataset, DataLoader
from hilbertcurve.hilbertcurve import HilbertCurve

from config import S3DISConfig


# Augmentation functions
def rotate_points(points):
    # scenter points at origin for rotation
    mins = points.max(axis=0) / 2.0
    orig_points = points - np.array([mins[0], mins[1], 0.0])

    # random rotation around vertical (z) axis
    theta = np.random.rand() * 2 * np.pi

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

    # rotate points and shift them back
    rot_points = np.matmul(R, orig_points.T).T + np.array([mins[0], mins[1], 0.0])

    return rot_points


def scale_points(points):
    scale_factor = 0.8 + np.random.rand(3) * 0.4  # random scale between 0.8 and 1.2
    # scale_factor[0] *= np.random.rand() * 2 - 1  # random symmetry around x only
    scale_factor[0] *= np.random.choice([-1, 1])    # I don't think we need shrink x. data distribution is not the same as testing
    return points * scale_factor


def add_noise_points(points):
    noise = np.random.normal(scale=0.001, size=points.shape)

    return points + noise


def get_hilbert_rotations(points, theta=np.pi):
    c, s = np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])
    Ry = np.array([[c, 0, s],
                   [0, 1, 0],
                   [-s, 0, c]])
    Rz = np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])

    shift = points[:, :3].min(axis=1) + (points[:, :3].max(axis=1) - points[:, :3].min(axis=1)) / 2.0
    data_shift = points[:, :3] - shift[:, np.newaxis]
    rot_points_x = np.matmul(Rx, data_shift.T).T
    rot_points_y = np.matmul(Ry, data_shift.T).T
    rot_points_z = np.matmul(Rz, data_shift.T).T

    return np.stack((data_shift, rot_points_x, rot_points_y, rot_points_z), axis=0)


class S3DISDataset(Dataset):
    def __init__(self, data_label, num_features=9, augment=False, p=7):
        self.augment = augment
        self.num_features = num_features
        self.data, self.label = data_label
        self.p = p
        # todo: find the best p.
        # compute hilbert order for voxelized space
        logging.info('Computing hilbert distances...')
        self.hilbert_curve = HilbertCurve(self.p, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        pointcloud = self.data[item, ...]
        label = self.label[item, ...]

        if self.augment:
            # todo: the data augmentation is only on the first 3 dims of the data.
            # todo: after scaling the data ranging is no longer 0, 1. However, in testing, what is the data range?
            # todo: pointcloud is [-0.5, 3.164]
            points = pointcloud[:, :3]
            points = rotate_points(points)
            points = scale_points(points)
            points = add_noise_points(points)
            pointcloud[:, :3] = points

        # for testing :
        # vis_points(pointcloud[:, 0:3], pointcloud[:, 3:6])
        # get coordinates
        coordinates = pointcloud[:, :3] - pointcloud[:, :3].min(axis=0)

        # compute hilbert order
        points_norm = pointcloud[:, :3] - pointcloud[:, :3].min(axis=0)
        points_norm /= points_norm.max(axis=0) + 1e-23

        # order points in hilbert order
        points_voxel = np.floor(points_norm * (2 ** self.p - 1))
        hilbert_dist = np.zeros(points_voxel.shape[0])
        for i in range(points_voxel.shape[0]):
            hilbert_dist[i] = self.hilbert_curve.distance_from_coordinates(points_voxel[i, :].astype(int))
        idx = np.argsort(hilbert_dist).copy()
        # todo: why whould idx change?
        pointcloud, coordinates, label = pointcloud[idx, :], coordinates[idx, :], label[idx]

        # return appropriate number of features
        if self.num_features == 4:
            # todo: why /255. gives better results?
            pointcloud = np.hstack((pointcloud[:, 3:6]/255., pointcloud[:, 2, np.newaxis]))
        elif self.num_features == 5:
            pointcloud = np.hstack((np.ones((pointcloud.shape[0], 1)), pointcloud[:, 3:6],
                                    pointcloud[:, 2, np.newaxis]))
        elif self.num_features == 9:
            min_val = pointcloud[:, :3].min(axis=0)
            pointcloud = np.hstack((pointcloud[:, :3] - min_val, pointcloud[:, 3:6], pointcloud[:, 6:9]))
        else:
            raise ValueError('Incorrect number of features provided. Values should be 4, 5, or 9, but {} provided'
                             .format(self.num_features))

        # return pointcloud[idx, :], coordinates[idx, :], label[idx]
        return pointcloud, coordinates, label


class S3DIS:
    """
    Class to represent data from S3DIS dataset, to perform
    semantic segmentation.
    """
    def __init__(self, args):
        """
        Initialize parameters of S3DIS class using the information
        provided in the Config object. More details about the
        configuration parameters in `config.py`

        :param Config: Config object (refer to `config.py`)
        """

        # Label names and colors for visualization:

        # dictionary with label values and names
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'chair',
                               8: 'table',
                               9: 'bookcase',
                               10: 'sofa',
                               11: 'board',
                               12: 'clutter'}

        # array mapping a label value to a color (used for visualizations)
        self.label_to_color = np.array([[0, 255, 0],
                                        [0, 0, 255],
                                        [0, 255, 255],
                                        [255, 255, 0],
                                        [255, 0, 255],
                                        [100, 100, 255],
                                        [200, 200, 100],
                                        [170, 120, 200],
                                        [255, 0, 0],
                                        [200, 100, 100],
                                        [10, 200, 100],
                                        [200, 200, 200],
                                        [50, 50, 50]])

        # dictionary with label names and corresponding colors
        self.name_to_color = {v: self.label_to_color[k] for k, v in self.label_to_names.items()}

        # Configuration parameters:
        self.config = self._create_config(args)

        # generate unique directory to save checkpoints
        self.experiment_dir = self._generate_experiment_directory()

        self.config.save(self.experiment_dir)

    @staticmethod
    def _create_config(args):
        config = S3DISConfig()

        if args.model is not None:
            config.model = args.model

        if args.backbone is not None:
            config.backbone = args.backbone

        if args.test_area is not None:
            config.test_area = args.test_area

        if args.kernel_size is not None:
            config.kernel_size = args.kernel_size

        if args.num_feats is not None:
            config.num_feats = args.num_feats

        if args.lr is not None:
            config.lr = args.lr

        if args.batch_size is not None:
            config.batch_size = int(args.batch_size)

        if args.augment is not None:
            config.augment = args.augment

        if args.bias is not None:
            config.bias = args.bias

        config.gpu_index = args.gpu
        config.multi_gpu = args.multi_gpu
        config.root_dir = args.root_dir
        config.model_dir = args.model_dir

        return config

    def _generate_experiment_directory(self):
        """
        Helper function to create checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
            Y-m-d_H:M_prefixStr_lr_batchSize_modelName_augmentation_numEpochs__UUID
        """
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        experiment_string = '{}_{}_A{}_Cin{}_lr{}_B{}_K{}_Sigma{}_p{}_{}_{}_{}_{}_Epo{}_{}' \
            .format(timestamp, self.config.dataset, self.config.test_area, self.config.num_feats, self.config.lr,
                    self.config.batch_size, self.config.kernel_size, self.config.sigma,  self.config.p, self.config.model,
                    self.config.backbone, 'augment' if self.config.augment else 'no-augment',
                    'bias' if self.config.bias else 'no-bias', self.config.max_epochs, uuid.uuid4())

        # experiment_dir = os.path.join(os.path.curdir, experiment_string)
        experiment_dir = os.path.join(self.config.model_dir, experiment_string)
        pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)

        return experiment_dir

    def _load_data(self):
        # first we load the room file list
        with open(os.path.join(self.config.root_dir, 'room_filelist.txt'), 'r') as f:
            room_list = np.array([line.rstrip() for line in f.readlines()])

        # load filenames
        with open(os.path.join(self.config.root_dir, 'all_files.txt'), 'r') as f:
            all_files = np.array([os.path.join(self.config.root_dir, os.path.basename(line.rstrip())) for line in f.readlines()])

        all_data = []
        all_label = []
        # for h5_name in tqdm(glob.glob(os.path.join(root_dir, 'ply_data_all*.h5')), desc='Loading all data'):
        for h5_name in tqdm(all_files, desc='Loading all data'):
            f = h5py.File(h5_name, 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()

            all_data.append(data)
            all_label.append(label)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        logging.info('Creating data splits for Area {}'.format(self.config.test_area))
        # now we split data into train or test based on test area
        test_idx = ['Area_{}'.format(self.config.test_area) in x for x in room_list]
        train_idx = ['Area_{}'.format(self.config.test_area) not in x for x in room_list]

        train_data = all_data[train_idx, ...]
        train_label = all_label[train_idx, ...]

        test_data = all_data[test_idx, ...]
        test_label = all_label[test_idx, ...]

        return train_data, train_label, test_data, test_label

    def get_dataloaders(self):
        train_data, train_label, test_data, test_label = self._load_data()

        datasets = {
            'train': S3DISDataset(data_label=(train_data, train_label),
                                  num_features=self.config.num_feats,
                                  augment=self.config.augment),
            'test': S3DISDataset(data_label=(test_data, test_label),
                                 num_features=self.config.num_feats,
                                 augment=False)
        }

        dataloaders = {x: DataLoader(dataset=datasets[x],
                                     batch_size=self.config.batch_size,
                                     num_workers=4,
                                     shuffle=(x == 'train'))
                       for x in ['train', 'test']}

        return dataloaders


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--root_dir', default='', type=str)
    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--test_area', default=5, type=int)
    parser.add_argument('--num_feats', default=5, type=int)
    parser.add_argument('--multi_gpu', default=False, action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--backbone', default=None, type=str)
    parser.add_argument('--kernel_size', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--augment', default=True, action='store_true')

    args = parser.parse_args()

    args.root_dir = '/media/thabetak/a5411846-373b-430e-99ac-01222eae60fd/S3DIS/indoor3d_sem_seg_hdf5_data'
    args.model_dir = '/media/thabetak/a5411846-373b-430e-99ac-01222eae60fd/3d_datasets/S3DIS/indoor3d_sem_seg_hdf5_data'

    dataset = S3DIS(args)
    dataloaders = dataset.get_dataloaders()

    for phase in ['train', 'test']:
        print(phase.upper())
        print('\tDataloder {}'.format(len(dataloaders[phase])))
        for i, (data, coords, seg_label) in enumerate(dataloaders[phase]):
            print('\tData {} Coords {} Seg Label {}'
                  .format(data.size(), coords.size(), seg_label.size()))
            if i >= 3:
                break
