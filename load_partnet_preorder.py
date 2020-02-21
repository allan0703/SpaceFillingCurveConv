from argparse import ArgumentParser
import numpy as np
import os
import os.path as osp
from tqdm import tqdm, trange
import h5py
import logging
import time
import glob
import uuid
import pathlib
from torch.utils.data import Dataset, DataLoader
from hilbertcurve.hilbertcurve import HilbertCurve

from config import PartnetConfig


# Augmentation functions
def rotate_points(points):
    # scenter points at origin for rotation
    mins = points.max(axis=0) / 2.0
    orig_points = points - np.array([mins[0], mins[1], 0.0])

    # random rotation around vertical (z) axis
    theta = np.random.rand() * 2 * np.pi

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])

    # rotate points and shift them back
    rot_points = np.matmul(R, orig_points.T).T + np.array([mins[0], mins[1], 0.0])

    return rot_points


def scale_points(points):
    scale_factor = 0.8 + np.random.rand(3) * 0.4  # random scale between 0.8 and 1.2
    scale_factor[0] *= np.random.rand() * 2 - 1  # random symmetry around x only

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


def get_edge_index(idx, k=9):
    # num_points : N
    n = len(idx)  # idx  SFC RGB+Z n-> num_points k->num_neighbors
    edge_index_index = np.zeros((n, k))
    for i in range(0, n):
        if k // 2 <= i < (n - (k - k // 2)):  # here 5 is 4 front neighbors and 5 behind neighbors
            edge_index_index[i] = np.concatenate((np.arange(i - k // 2, i), np.arange(i + 1, i + (k - k // 2) + 1)))
        elif i < k // 2:
            l = np.arange(k + 1)
            l = filter(lambda x: x != i, l)
            l = [j for j in l]
            edge_index_index[i] = l
        else:
            l = np.arange(n - k - 1, n)
            l = filter(lambda x: x != i, l)
            l = [j for j in l]
            edge_index_index[i] = l
    center_point = np.arange(n).transpose()
    center_point = np.repeat(center_point[:, np.newaxis], k, axis=1)
    edge_index = np.stack((idx[center_point.astype(np.int16)], idx[edge_index_index.astype(np.int16)]), axis=0)

    edge_index = edge_index[:, edge_index[0, :, 1].argsort(), :]
    return edge_index


def dgcnn_augment(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    np.random.shuffle(translated_pointcloud)
    return translated_pointcloud


class PartnetDataset(Dataset):
    def __init__(self, data_label, num_features=9, augment=False, sfc_neighbors=9, use_rotation=False):
        self.augment = augment
        self.num_features = num_features
        self.data, self.label = data_label
        self.p = 7
        self.p2 = 3
        self.neighbors = sfc_neighbors
        # self.edge_index = get_edge_index_index(self.data.shape[1], sfc_neighbors)
        self.use_rotation = use_rotation

        # compute hilbert order for voxelized space
        logging.info('Computing hilbert distances...')
        self.hilbert_curve = HilbertCurve(self.p, 3)
        self.hilbert_curve_rgbz = HilbertCurve(self.p2, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        pointcloud = self.data[item, ...]
        label = self.label[item, ...]

        if self.augment:
            pointcloud = dgcnn_augment(pointcloud)

        coordinates = pointcloud[:, :3] - pointcloud[:, :3].min(axis=0)
        points_norm = pointcloud - pointcloud.min(axis=0)
        points_norm /= points_norm.max(axis=0) + 1e-23
        points_voxel = np.floor(points_norm * (2 ** self.p - 1))
        hilbert_dist = np.zeros(points_voxel.shape[0])
        for i in range(points_voxel.shape[0]):
            hilbert_dist[i] = self.hilbert_curve.distance_from_coordinates(points_voxel[i, :].astype(int))
        idx = np.argsort(hilbert_dist)
        pointcloud, coordinates = pointcloud[idx, :], coordinates[idx, :]

        # SFC 2 neighbors, by rotating the object
        rotation_z = np.transpose([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        points_voxel_rotation = np.matmul(pointcloud, rotation_z.T)
        points_norm = points_voxel_rotation - points_voxel_rotation.min(axis=0)
        points_norm /= points_norm.max(axis=0) + 1e-23
        points_voxel = np.floor(points_norm * (2 ** self.p2 - 1))
        for i in range(points_voxel.shape[0]):
            hilbert_dist[i] = self.hilbert_curve_rgbz.distance_from_coordinates(points_voxel[i, :].astype(int))
        idx2 = np.argsort(hilbert_dist)
        neighbors_edge_index = get_edge_index(idx2, self.neighbors)
        return pointcloud, coordinates, label, neighbors_edge_index


class Partnet:
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
        self.label_to_names = {0: 'Bag',
                               1: 'Bed',
                               2: 'Bottle',
                               3: 'Bowl',
                               4: 'Chair',
                               5: 'Clock',
                               6: 'Dishwasher',
                               7: 'Display',
                               8: 'Door',
                               9: 'Earphone',
                               10: 'Faucet',
                               11: 'Hat',
                               12: 'Keyboard',
                               13: 'Knife',
                               14: 'Lamp',
                               15: 'Laptop',
                               16: 'Microwave',
                               17: 'Mug',
                               18: 'Refrigerator',
                               19: 'Scissors',
                               20: 'StorageFurniture',
                               21: 'Table',
                               22: 'TrashCan',
                               23: 'Vase'
                               }

        # Configuration parameters:
        self.config = self._create_config(args)

        # generate unique directory to save checkpoints
        self.experiment_dir = self._generate_experiment_directory()

        self.config.save(self.experiment_dir)

    @staticmethod
    def _create_config(args):
        config = PartnetConfig()
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

        if args.category is not None:
            config.category = args.category

        if args.level is not None:
            config.level = args.level

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
        experiment_string = '{}_{}_{}_{}_Cin{}_lr{}_B{}_K{}_Sigma{}_{}_{}_Epo{}_{}' \
            .format(timestamp, self.config.dataset, self.config.model, self.config.backbone,
                    self.config.num_feats, self.config.lr, self.config.batch_size, self.config.kernel_size,
                    self.config.sigma,  'augment' if self.config.augment else 'no-augment',
                    'bias' if self.config.bias else 'no-bias', self.config.max_epochs, uuid.uuid4())
        experiment_dir = os.path.join(self.config.model_dir, experiment_string)
        pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)

        return experiment_dir

    def _load_data(self):
        data = {'train_data': [], 'train_label': [], 'val_data': [], 'val_label': [],
                'test_data': [], 'test_label': []}
        # --------------------------------------------------------------------------------------------------------------
        #  for instance segmentation
        #
        #     if self.config.dataset == 'ins_seg_h5':
        #         raw_path = osp.join(self.config.raw_dir, dataset)
        #         categories = glob(osp.join(raw_path, '*'))
        #         categories = sorted([x.split(os.sep)[-1] for x in categories])
        #
        #         data_list = []
        #         for target, category in enumerate(tqdm(categories)):
        #             folder = osp.join(raw_path, category)
        #             paths = glob('{}/{}-*.h5'.format(folder, dataset))
        #             labels, nors, opacitys, pts, rgbs = [], [], [], [], []
        #             for path in paths:
        #                 with h5py.File(path, 'r') as f:
        #                     pts += f['pts']
        #                     labels += f['label']
        #                     nors += f['nor']
        #                     opacitys += f['opacity']
        #                     rgbs += f['rgb']
        #             pointcloud = np.concatenate((np.stack(pts), np.stack(rgbs) / 255.,
        #                                          np.stack(nors), np.stack(opacitys)), dim=1)
        #             lable = np.stack(labels, dim=0)
        #             data[dataset+'_data'] = pointcloud
        #             data[dataset+'_label'] = lable
        #
        #     else:
        # --------------------------------------------------------------------------------------------------------------

        object = '-'.join([self.config.category, str(self.config.level)])
        raw_path = osp.join(self.config.root_dir, self.config.dataset)  #dataset is sem_seg_h5
        categories = glob.glob(osp.join(raw_path, object))
        categories = sorted([x.split(os.sep)[-1] for x in categories])
        for target, category in enumerate(tqdm(categories)):
            folder = osp.join(raw_path, category)
            for dataset_split in ['train', 'val', 'test']:
                paths = glob.glob('{}/{}-*.h5'.format(folder, dataset_split))
                labels, pts = [], []
                for path in paths:
                    with h5py.File(path, 'r') as f:
                        pts += f['data']
                        labels += f['label_seg']  # todo: dype long,.
                pointcloud = np.stack(pts)
                lable = np.stack(labels)
                data[dataset_split + '_data'] = pointcloud
                data[dataset_split + '_label'] = lable
        return data['train_data'], data['train_label'], data['val_data'], data['val_label'], data['test_data'], data[
            'test_label']

    def get_dataloaders(self):
        train_data, train_label, val_data, val_lable, test_data, test_label = self._load_data()
        datasets = {
            'train': PartnetDataset(data_label=(train_data, train_label),
                                    num_features=self.config.num_feats,
                                    augment=self.config.augment),
            'val': PartnetDataset(data_label=(val_data, val_lable), num_features=self.config.num_feats,
                                  augment=False),
            'test': PartnetDataset(data_label=(test_data, test_label), num_features=self.config.num_feats,
                                   augment=False)
        }

        dataloaders = {x: DataLoader(dataset=datasets[x],
                                     batch_size=self.config.batch_size,
                                     num_workers=4,
                                     shuffle=(x == 'train'))
                       for x in self.config.phases}

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
    parser.add_argument('--loglevel', default='INFO', type=str, help='logging level')
    parser.add_argument('--category', default='Bed', type=str, help='Area used for test set (1, 2, 3, 4, 5, or 6)')
    parser.add_argument('--level', default=3, type=int, help='Area used for test set (1, 2, 3, 4, 5, or 6)')


    args = parser.parse_args()

    args.root_dir = '/home/wangh0j/data/sfc/'
    args.model_dir = '/home/wangh0j/SFC-Convs/log/'

    dataset = Partnet(args)
    dataloaders = dataset.get_dataloaders()

    for phase in ['train', 'test']:
        print(phase.upper())
        print('\tDataloder {}'.format(len(dataloaders[phase])))
        for i, (data, coords, seg_label, e) in enumerate(dataloaders[phase]):
            print('\tData {} Coords {} Seg Label {}'
                  .format(data.size(), coords.size(), seg_label.size(), e.size()))
            if i >= 3:
                break
