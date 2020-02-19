from argparse import ArgumentParser
import numpy as np
import os
from tqdm import tqdm, trange
import h5py
import logging
import time
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
        self.hilbert_curve_rgbz = HilbertCurve(self.p2, 6)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        pointcloud = self.data[item, ...]
        label = self.label[item, ...]

        if self.augment:
            points = pointcloud[:, :3]
            points = rotate_points(points)
            points = scale_points(points)
            points = add_noise_points(points)

            pointcloud[:, :3] = points

        # get coordinates
        coordinates = pointcloud[:, :3] - pointcloud[:, :3].min(axis=0)
        # z_rgb = np.concatenate((pointcloud[:, 2], pointcloud[:, 3:6]), axis=1)

        # compute hilbert order
        # here. We build a SFC-Curve by using the XYZ. norm the xyz
        points_norm = pointcloud[:, :3] - pointcloud[:, :3].min(axis=0)
        points_norm /= points_norm.max(axis=0) + 1e-23
        # z_rgb_norm = np.concatenate((points_norm[:, 2, np.newaxis], pointcloud[:, 3:6]), axis=1)
        # order points in hilbert order
        points_voxel = np.floor(points_norm * (2 ** self.p - 1))

        hilbert_dist = np.zeros(points_voxel.shape[0])
        for i in range(points_voxel.shape[0]):
            hilbert_dist[i] = self.hilbert_curve.distance_from_coordinates(points_voxel[i, :].astype(int))
        idx = np.argsort(hilbert_dist)  # index by using xyz

        pointcloud, coordinates, label = pointcloud[idx, :], coordinates[idx, :], label[idx]
        points_norm, points_voxel = points_norm[idx, :], points_voxel[idx, :]

        if self.use_rotation:
            rotation_z = np.transpose([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            points_voxel_rotation = np.matmul(points_voxel, rotation_z).astype(int)
            points_voxel_rotation[:, 0:2] += (2 ** self.p - 1)
            for i in range(points_voxel.shape[0]):
                hilbert_dist[i] = self.hilbert_curve.distance_from_coordinates(points_voxel_rotation[i, :].astype(int))
            idx2 = np.argsort(hilbert_dist)
        else:
            xyz_rgb_norm = np.concatenate((points_norm, pointcloud[:, 3:6]), axis=1)
            points_voxel1 = np.floor(xyz_rgb_norm * (2 ** self.p2 - 1))
            for i in range(points_voxel1.shape[0]):
                hilbert_dist[i] = self.hilbert_curve_rgbz.distance_from_coordinates(points_voxel1[i, :].astype(int))
            idx2 = np.argsort(hilbert_dist)

        neighbors_edge_index = get_edge_index(idx2, self.neighbors)

        # return appropriate number of features
        if self.num_features == 4:
            pointcloud = np.hstack((pointcloud[:, 3:6] / 255., pointcloud[:, 2, np.newaxis]))
        elif self.num_features == 5:
            pointcloud = np.hstack((np.ones((pointcloud.shape[0], 1)), pointcloud[:, 3:6],
                                    pointcloud[:, 2, np.newaxis]))
        elif self.num_features == 9:
            min_val = pointcloud[:, :3].min(axis=0)
            pointcloud = np.hstack((pointcloud[:, :3] - min_val, pointcloud[:, 3:6], pointcloud[:, 6:9]))
        else:
            raise ValueError('Incorrect number of features provided. Values should be 4, 5, or 9, but {} provided'
                             .format(self.num_features))
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

        object = '-'.join([self.config.obj_category, str(self.config.level)])
        raw_path = osp.join(self.raw_dir, self.config.dataset)
        categories = glob(osp.join(raw_path, object))
        categories = sorted([x.split(os.sep)[-1] for x in categories])
        for target, category in enumerate(tqdm(categories)):
            folder = osp.join(raw_path, category)
            for dataset_split in ['train', 'val', 'test']:
                paths = glob('{}/{}-*.h5'.format(folder, dataset_split))
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

    args = parser.parse_args()

    args.root_dir = '/home/wangh0j/data/sfc/S3DIS/raw'
    args.model_dir = '/home/wangh0j/SFC-Convs/log/'

    dataset = S3DIS(args)
    dataloaders = dataset.get_dataloaders()

    for phase in ['train', 'test']:
        print(phase.upper())
        print('\tDataloder {}'.format(len(dataloaders[phase])))
        for i, (data, coords, seg_label, e) in enumerate(dataloaders[phase]):
            print('\tData {} Coords {} Seg Label {}'
                  .format(data.size(), coords.size(), seg_label.size(), e.size()))
            if i >= 3:
                break


class PartNet(InMemoryDataset):
    r"""The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
    the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
    <http://buildingparser.stanford.edu/images/3D_Semantic_Parsing.pdf>`_
    paper, containing point clouds of six large-scale indoor parts in three
    buildings with 12 semantic elements (and one clutter class).

    Args:
        root (string): Root directory where the dataset should be saved.
        dataset (str, optional): Which dataset to use (ins_seg_h5, or sem_seg_h5).
            (default: :obj:`ins_seg_h5`)
        phase (str, optional): If :obj:`test`, loads the testing dataset,
            If :obj:`val`, loads the validation dataset,
            otherwise the training dataset. (default: :obj:`train`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    # clss = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',  # 0-9
    #         'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator', 'Scissors',
    #         # 10-19
    #         'StorageFurniture', 'Table', 'TrashCan', 'Vase']  # 20-23
    #
    # url = ('https://shapenet.cs.stanford.edu/media/'
    #        'indoor3d_sem_seg_hdf5_data.zip')

    def __init__(self,
                 root,
                 dataset='sem_seg_h5',
                 obj_category='Bed',
                 level=3,
                 phase='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 hilbert_order=True,
                 hilbert_level=7):
        self.dataset = dataset
        self.level = level
        self.obj_category = obj_category
        self.object = '-'.join([self.obj_category, str(self.level)])
        self.level_folder = 'level_' + str(self.level)
        self.processed_file_folder = osp.join(self.dataset, self.level_folder, self.object)
        self.p = hilbert_level
        self.hilbert_order = hilbert_order
        if self.hilbert_order:
            self.hilbert_curve = HilbertCurve(self.p, 3)

        super(PartNet, self).__init__(root, transform, pre_transform, pre_filter)
        if phase == 'test':
            path = self.processed_paths[1]
        elif phase == 'val':
            path = self.processed_paths[2]
        else:
            path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [self.dataset]

    @property
    def processed_file_names(self):
        return osp.join(self.processed_file_folder, 'train.pt'), osp.join(self.processed_file_folder, 'test.pt'), \
               osp.join(self.processed_file_folder, 'val.pt')

    def download(self):
        path = osp.join(self.raw_dir, self.dataset)
        if not osp.exists(path):
            raise FileExistsError('PartNet can only downloaded via application. '
                                  'See details in https://cs.stanford.edu/~kaichun/partnet/')
        # path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split(os.sep)[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self):
        # save to processed_paths
        processed_path = osp.join(self.processed_dir, self.processed_file_folder)
        if not osp.exists(processed_path):
            os.makedirs(osp.join(processed_path))
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])
        torch.save(self.process_set('val'), self.processed_paths[2])

    def process_set(self, dataset):
        if self.dataset == 'ins_seg_h5':
            raw_path = osp.join(self.raw_dir, 'ins_seg_h5_for_sgpn', self.dataset)
            categories = glob(osp.join(raw_path, '*'))
            categories = sorted([x.split(os.sep)[-1] for x in categories])

            data_list = []
            for target, category in enumerate(tqdm(categories)):
                folder = osp.join(raw_path, category)
                paths = glob('{}/{}-*.h5'.format(folder, dataset))
                labels, nors, opacitys, pts, rgbs = [], [], [], [], []
                for path in paths:
                    with h5py.File(path, 'r') as f:
                        pts += torch.from_numpy(f['pts'][:]).unbind(0)
                        labels += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)
                        nors += torch.from_numpy(f['nor'][:]).unbind(0)
                        opacitys += torch.from_numpy(f['opacity'][:]).unbind(0)
                        rgbs += torch.from_numpy(f['rgb'][:]).to(torch.float32).unbind(0)

                for i, (pt, label, nor, opacity, rgb) in enumerate(zip(pts, labels, nors, opacitys, rgbs)):
                    data = Data(pos=pt[:, :3], y=label, norm=nor[:, :3],
                                x=torch.cat((opacity.unsqueeze(-1), rgb / 255.), 1))

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    pos, x, y, norm = data.pos, data.x, data.y, data.norm
                    point = (pos.numpy() + 1) / 2
                    points_voxel = np.floor(point * (2 ** self.p - 1)).astype(int)
                    hilbert_dist = np.zeros(points_voxel.shape[0])

                    for point_idx in range(points_voxel.shape[0]):
                        hilbert_dist[point_idx] = self.hilbert_curve.distance_from_coordinates(
                            points_voxel[point_idx, :])
                    idx = np.argsort(hilbert_dist)
                    pos, x, norm, y = pos[idx, :], x[idx, :], norm[idx, :], y[idx]
                    data = Data(pos=pos, x=x, y=y, norm=norm)
                    data_list.append(data)
        else:
            raw_path = osp.join(self.raw_dir, self.dataset)
            categories = glob(osp.join(raw_path, self.object))
            categories = sorted([x.split(os.sep)[-1] for x in categories])
            data_list = []
            # class_name = []
            for target, category in enumerate(tqdm(categories)):
                folder = osp.join(raw_path, category)
                paths = glob('{}/{}-*.h5'.format(folder, dataset))
                labels, pts = [], []
                # clss = category.split('-')[0]

                for path in paths:
                    f = h5py.File(path)
                    pts += torch.from_numpy(f['data'][:].astype(np.float32)).unbind(0)
                    labels += torch.from_numpy(f['label_seg'][:].astype(np.float32)).to(torch.long).unbind(0)
                for i, (pt, label) in enumerate(zip(pts, labels)):
                    data = Data(pos=pt[:, :3], y=label)
                    # data = PartData(pos=pt[:, :3], y=label, clss=clss)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if self.hilbert_order:
                        # order points in hilbert order
                        # after normalizaiton, -1 to 1
                        pos, y = data.pos, data.y
                        point = (pos.numpy() + 1) / 2  # normalize to 0, 1
                        points_voxel = np.floor(point * (2 ** self.p - 1)).astype(int)
                        hilbert_dist = np.zeros(points_voxel.shape[0])

                        # todo: we want to try two methods.
                        # todo: 1. globally along with locally hilbert curve
                        # todo: 2. multi-level of hilbert curve
                        for point_idx in range(points_voxel.shape[0]):
                            hilbert_dist[point_idx] = self.hilbert_curve.distance_from_coordinates(
                                points_voxel[point_idx, :])
                        idx = np.argsort(hilbert_dist)
                        pos = pos[idx, :]
                        y = y[idx]
                        data = Data(pos=pos, y=y)
                    data_list.append(data)
        return self.collate(data_list)


def get_partnet_dataloaders(root_dir, phases, batch_size, dataset='sem_seg_h5', category='Bed', level=3, augment=False):
    """
    Create Dataset and Dataloader classes of the S3DIS dataset, for
    the phases required (`train`, `test`).

    :param root_dir: Directory with the h5 files
    :param phases: List of phases. Should be from {`train`, `test`}
    :param batch_size: Batch size
    :param category: Area used for test set (1, 2, 3, 4, 5, or 6)

    :return: 2 dictionaries, each containing Dataset or Dataloader for all phases
    """
    datasets = {
        'train': PartNet(root_dir, dataset, category, level, 'train'),
        'val': PartNet(root_dir, dataset, category, level, 'val'),
        'test': PartNet(root_dir, dataset, category, level, 'test'),

    }

    dataloaders = {x: DenseDataLoader(datasets[x], batch_size=batch_size, num_workers=4, shuffle=(x == 'train'))
                   for x in phases}
    return datasets, dataloaders, datasets['train'].num_classes


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a point cloud classification network using 1D convs and hilbert order.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, default='/data/sfc/partnet',
                        help='root directory containing PartNet data')
    args = parser.parse_args()

    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    root_dir = args.root_dir
    phases = ['train', 'val', 'test']
    batch_size = 8
    category = 5

    # verify hilbert order.
    p = 7
    N = 2
    hilbert_curve = HilbertCurve(p, N)
    for coords in [[0, 0], [0, 1], [1, 1], [1, 0]]:
        dist = hilbert_curve.distance_from_coordinates(coords)
        print(f'distance(x={coords}) = {dist}')
    #
    datasets, dataloaders, num_classes = get_partnet_dataloaders(root_dir=root_dir,
                                                                 phases=phases,
                                                                 batch_size=batch_size,
                                                                 category=category)

    for phase in phases:
        print(phase.upper())
        print('\tDataset {} Dataloder {}'.format(len(datasets[phase]), len(dataloaders[phase])))
        for i, data in enumerate(dataloaders[phase]):
            print(data.pos.shape)
            x = torch.cat((data.pos, data.x), dim=2).transpose(1, 2)
            seg_label = data.y
            print('\tData {} Seg Label {}'.format(x.size(), seg_label.size()))
            if i >= 3:
                break

    print('Test')
    print('\tDataset {} Dataloder {} Num classes {}'.format(len(datasets['test']),
                                                            len(dataloaders['test']), num_classes))
