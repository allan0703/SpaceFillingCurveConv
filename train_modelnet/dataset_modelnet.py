from argparse import ArgumentParser
import numpy as np
import os
from tqdm import tqdm
import glob
import h5py
import logging
import time
import uuid
import pathlib

from config import S3DISConfig
from torch.utils.data import Dataset, DataLoader
from hilbertcurve.hilbertcurve import HilbertCurve


def load_data(root_dir, phase):
    """
    Load ModelNet40 data from h5 files

    :param root_dir: Directory with Modelnet40 h5 data
    :param phase: `train` or `test`

    :return: 2 Numpy arrays for data (PxNx3) and labels (Px1)
    """
    all_data = []
    all_label = []
    for h5_name in tqdm(glob.glob(os.path.join(root_dir, 'ply_data_{}*.h5'.format(phase))),
                        desc='Loading data for phase {}'.format(phase)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')

        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return np.nan_to_num(all_data), all_label


def translate_pointcloud(pointcloud):
    """
    Augment pointcloud using uniform translations

    :param pointcloud: Nx3 array of points
    :return: Translated points (Nx3)
    """
    trans = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_data = pointcloud + trans
    return translated_data.astype('float32')


def rotate_pointcloud(pointcloud):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    theta = np.random.rand() * 2 * np.pi

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    rotated_data = np.matmul(pointcloud, R.T)
    return rotated_data.astype('float32')


def scale_pointcloud(pointcloud):
    """ Randomly scale a pointcloud
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(low=2./3., high=3./2., size=[3])
    points_scaled = pointcloud * scale
    return points_scaled.astype('float32')


def get_edge_index(idx, k=9):
    # num_points : N
    n = len(idx)  # idx  SFC RGB+Z n-> num_points k->num_neighbors
    edge_index_index = np.zeros((n, k))
    for i in range(0, n):
        if k // 2 <= i < (n - (k - k//2)):  # here 5 is 4 front neighbors and 5 behind neighbors
            edge_index_index[i] = np.concatenate((np.arange(i - k//2, i), np.arange(i + 1, i + (k - k//2) + 1)))
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


def shear_pointcloud(pointcloud):
    """ Randomly shear a pointcloud
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, sheared batch of point clouds
    """
    T = np.eye(3) + 0.1 * np.random.randn(3, 3)
    points_sheared = np.dot(pointcloud, T)

    return points_sheared.astype('float32')


def dgcnn_augment(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    np.random.shuffle(translated_pointcloud)
    return translated_pointcloud


class ModelNet40(Dataset):
    def __init__(self, data, label, augment=False, num_features=9, sfc_neighbors=9, num_points=1024):
        self.augment = augment
        self.data, self.label = data, label
        self.num_features = num_features
        self.num_points = num_points
        self.neighbors = sfc_neighbors
        self.p = 7  # hilbert iteration
        self.p2 = 3

        # by changing the value of p, we can control the level of hilbert curve.
        # this hyperparameter has to be careful and ideally, p should be different for each point cloud.
        # (because the density distribution is different

        # compute hilbert order for voxelized space
        logging.info('Computing hilbert distances...')
        self.hilbert_curve = HilbertCurve(self.p, 3)
        self.hilbert_curve_rgbz = HilbertCurve(self.p2, 3)
        # different from voxelization, we are much more efficient

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        pointcloud = self.data[item, :self.num_points]  # todo: only have xyz??
        label = self.label[item]

        # augment data when requested
        if self.augment:
            # pointcloud = dgcnn_augment(pointcloud)
            pointcloud = scale_pointcloud(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = shear_pointcloud(pointcloud)
        # normalize points

        # todo: some problems with neighboors.
        coordinates = pointcloud[:, :3] - pointcloud[:, :3].min(axis=0)
        points_norm = pointcloud - pointcloud.min(axis=0)
        points_norm /= points_norm.max(axis=0) + 1e-23
        # order points in hilbert order
        points_voxel = np.floor(points_norm * (2 ** self.p - 1))
        hilbert_dist = np.zeros(points_voxel.shape[0])

        for i in range(points_voxel.shape[0]):
            hilbert_dist[i] = self.hilbert_curve.distance_from_coordinates(points_voxel[i, :].astype(int))
        idx = np.argsort(hilbert_dist)
        pointcloud, coordinates = pointcloud[idx, :], coordinates[idx, :]
        points_norm, points_voxel = points_norm[idx, :], points_voxel[idx, :]

        # if self.use_rotation:
        rotation_z = np.transpose([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        points_voxel_rotation = np.matmul(points_voxel, rotation_z).astype(int)
        points_voxel_rotation[:, 0:2] += (2 ** self.p2 - 1)
        for i in range(points_voxel.shape[0]):
            hilbert_dist[i] = self.hilbert_curve_rgbz.distance_from_coordinates(points_voxel_rotation[i, :].astype(int))
        idx2 = np.argsort(hilbert_dist)
        # else:
        #     pointcloud1 = rotate_pointcloud(pointcloud)
        #     points_norm1 = pointcloud1 - pointcloud1.min(axis=0)
        #     points_norm1 /= points_norm1.max(axis=0) + 1e-23
        #     xyz_rgb_norm = points_norm1  # np.concatenate((points_norm, pointcloud[:, 3:6]), axis=1)
        #     points_voxel1 = np.floor(xyz_rgb_norm * (2 ** self.p2 - 1))
        #     for i in range(points_voxel1.shape[0]):
        #         hilbert_dist[i] = self.hilbert_curve_rgbz.distance_from_coordinates(points_voxel1[i, :].astype(int))
        #     idx2 = np.argsort(hilbert_dist)
        neighbors_edge_index = get_edge_index(idx2, self.neighbors)
        return pointcloud, coordinates, label, neighbors_edge_index


class ModelNet:
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
        experiment_string = '{}_{}_Cin{}_lr{}_B{}_K{}_Sigma{}_{}_{}_{}_{}_Epo{}_{}' \
            .format(timestamp, self.config.dataset,  self.config.num_feats, self.config.lr,
                    self.config.batch_size, self.config.kernel_size, self.config.sigma,  self.config.model,
                    self.config.backbone, 'augment' if self.config.augment else 'no-augment',
                    'bias' if self.config.bias else 'no-bias', self.config.max_epochs, uuid.uuid4())

        # experiment_dir = os.path.join(os.path.curdir, experiment_string)
        experiment_dir = os.path.join(self.config.model_dir, experiment_string)
        pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)

        return experiment_dir

    def get_dataloaders(self):
        train_data, train_label = load_data(root_dir=self.config.root_dir, phase='train')
        test_data, test_label = load_data(root_dir=self.config.root_dir, phase='test')
        """
        Create Dataset and Dataloader classes of the Modelnet40 dataset, for
        the phases required (`train`, `test`). Dataset can be downloaded from:

            - https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

        :param root_dir: Directory with `modelnet40_ply_hdf5_2048` folder
        :param phases: List of phases. Should be from {`train`, `test`}
        :param batch_size: Batch size
        :param augment: If True then we use data augmentation on training
        :return: 2 dictionaries, each containing Dataset or Dataloader for all phases
        """
        datasets = {
            'train': ModelNet40(train_data, train_label,
                                num_features=self.config.num_feats,
                                num_points=self.config.num_points,
                                # sfc_neighbors=self.config.k # knn neighboors
                                augment=self.config.augment),
            'test': ModelNet40(test_data, test_label,
                               num_features=self.config.num_feats,
                               num_points=self.config.num_points,
                               augment=False)
        }
        dataloaders = {x: DataLoader(dataset=datasets[x],
                                     batch_size=self.config.batch_size,
                                     num_workers=4,
                                     shuffle=(x == 'train'))
                       for x in ['train', 'test']}

        return dataloaders


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

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

    args.root_dir = '/home/wangh0j/data/sfc/modelnet40_ply_hdf5_2048'
    args.model_dir = '/home/wangh0j/SFC-Convs/log/'

    datasets = ModelNet(args)
    dataloaders = datasets.get_dataloaders()

    # root_dir = '/home/wangh0j/data/sfc/modelnet40_ply_hdf5_2048'
    phases = ['train', 'test']
    # batch_size = 32
    # augment = False
    #
    # datasets, dataloaders = get_modelnet40_dataloaders(root_dir, phases, batch_size, augment)

    for phase in phases:
        print(phase.upper())
        print('\t Dataloder {}'.format(len(dataloaders[phase])))
        for i, (data, coords, seg_label, e) in enumerate(dataloaders[phase]):
            print('\tData {} Label {} seg_label{} edge_index{} '.format(data.size(), coords.size(), seg_label.size(), e.size()))
            if i >= 3:
                break

    print('Test')
    print('\t Dataloder {}'.format(len(dataloaders['test'])))
