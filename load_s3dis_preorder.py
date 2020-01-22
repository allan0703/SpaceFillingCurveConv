import os
import os.path as osp
import shutil
from tqdm import tqdm
import h5py
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url, extract_zip, DenseDataLoader)
import torch_geometric.transforms as T
import logging
import numpy as np
from torch.utils.data import DataLoader
from hilbertcurve.hilbertcurve import HilbertCurve
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

class S3DIS(InMemoryDataset):
    r"""The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
    the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
    <http://buildingparser.stanford.edu/images/3D_Semantic_Parsing.pdf>`_
    paper, containing point clouds of six large-scale indoor parts in three
    buildings with 12 semantic elements (and one clutter class).

    Args:
        root (string): Root directory where the dataset should be saved.
        test_area (int, optional): Which area to use for testing (1-6).
            (default: :obj:`6`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
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

    url = ('https://shapenet.cs.stanford.edu/media/'
           'indoor3d_sem_seg_hdf5_data.zip')

    def __init__(self,
                 root,
                 test_area=5,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 hilbert_order=True,
                 hilbert_level=7):
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
        self.p = hilbert_level
        self.hilbert_order = hilbert_order
        if self.hilbert_order:
            self.hilbert_curve = HilbertCurve(self.p, 3)
        super(S3DIS, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['all_files.txt', 'room_filelist.txt']

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return ['{}_{}.pt'.format(s, test_area) for s in ['train', 'test']]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split(os.sep)[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        with open(self.raw_paths[1], 'r') as f:
            rooms = f.read().split('\n')[:-1]

        xs, ys = [], []
        for filename in filenames:
            f = h5py.File(osp.join(self.raw_dir, filename))
            # todo: check the data range
            xs += torch.from_numpy(f['data'][:]).unbind(0) #features
            ys += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)

        test_area = 'Area_{}'.format(self.test_area)
        train_data_list, test_data_list = [], []
        for i, (x, y) in enumerate(tqdm(zip(xs, ys), total=len(xs))):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)  #x y z coordiante

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Hilbert curve
            if self.hilbert_order:
                # order points in hilbert order
                # after normalizaiton, -1 to 1
                pos, x, y = data.pos, data.x, data.y  # pos x y are tensors
                multi_pos, multi_x, multi_y = [],[],[]
                # multi_pos = torch.empty_like(pos)
                # multi_x = torch.empty_like(x)
                # multi_y = torch.empty_like(y)
                point = (pos.numpy()+1)/2  # normalize to 0, 1
                points_voxel = np.floor(point * (2 ** self.p - 1)).astype(int) # what's the meaning  n*3  in range(1 - 2**p)
                hilbert_dist = np.zeros(points_voxel.shape[0])

                rotation_x = np.transpose([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # rotate along x coordinate
                rotation_y = np.transpose([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                rotation_z = np.transpose([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                rotation_matrices = [np.eye(3), rotation_x, rotation_y, rotation_z]

                # todo: we want to try two methods.
                # todo: 1. globally along with locally hilbert curve
                # todo: 2. multi-level of hilbert curve
                #points_voxel_rotation = np.matmul(points_voxel, rotation_matrices) # n * 12

                for num_rotation in range(4):
                    points_voxel_rotation = np.matmul(points_voxel, rotation_matrices[num_rotation]).astype(int)  # 4*n * 3
                    if num_rotation:
                        points_voxel_rotation[:, num_rotation % 3] += (2 ** self.p - 1) #ping
                    # step1: rotate point cloud
                    # loop to calulate hilbert_dist
                    # put pos, x, y  into a list
                    #multi_pos.append(pos[idx, :])
                    # pos = torch.stack(multi_pos, dim=0)
                    # data = Data(pos=pos, x=x, y=y)
                    for point_idx in range(points_voxel_rotation.shape[0]):
                        hilbert_dist[point_idx] = self.hilbert_curve.distance_from_coordinates(points_voxel_rotation[point_idx, :3])  #want hilbert_dist = n*4

                    idx = np.argsort(hilbert_dist) #fanhui cong da daoxiao de suoyi

                    multi_pos.append(pos[idx, :])
                    multi_x.append(x[idx, :])
                    if num_rotation == 0:
                        multi_y = y[idx]
                    # multi_y.append(y[idx])
                    # if num_rotation == 0:
                    #     multi_pos = pos[idx,:]
                    #     multi_x = x[idx, :]
                    #     multi_y = y[idx]
                    # else:
                    #     multi_pos = torch.stack((multi_pos, pos[idx, :]), dim=0)
                    #     multi_x = torch.stack((multi_x, x[idx, :]), dim=0)
                    #     multi_y = torch.stack((multi_y, y[idx]),  dim=0) #4*n*c

                # multi_pos = pos[idx,:]
                # multi_x = x[idx, :]
                # multi_y = y[idx]

                #     multi_pos.append(pos[idx, :]) # get N_0 * 3
                #         # pos = torch.stack(multi_pos, dim=0)
                #     multi_x.append(x[idx, :])
                #     multi_y.append(y[idx])
                multi_pos = torch.stack(multi_pos, dim=0)
                multi_x = torch.stack(multi_x, dim=0)
                # multi_y = torch.stack(multi_y, dim=0)

                multi_pos = multi_pos.permute(2,1,0) # yinggai 4 * n *d haishi n*d*4  bixudeishizheyang yinggai
                multi_x = multi_x.permute(2,1,0)
                # multi_y = multi_y.permute(1,0)
                data = Data(pos=multi_pos, x=multi_x, y=multi_y) # get 4N* dimension

            if test_area not in rooms[i]:
                train_data_list.append(data)
            else:
                test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0]) # data.pos = 4*4096*3
        torch.save(self.collate(test_data_list), self.processed_paths[1])


def dataToCsv(file, data, columns):
    data = list(data)
    columns = list(columns)
    file_data = pd.DataFrame(data, index=range(len(data)), columns=columns)
    # file_target = pd.DataFrame(target, index=range(len(data)), columns=['target'])
    # file_all = file_data.join(file_target, how='outer')
    file_data.to_csv(file)

def get_s3dis_dataloaders(root_dir, phases, batch_size, category=5, augment=False):
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
        'train': S3DIS(root_dir, category, True, pre_transform=T.NormalizeScale()),
        'test': S3DIS(root_dir, category, False, pre_transform=T.NormalizeScale())
    }

    dataloaders = {x: DenseDataLoader(datasets[x], batch_size=batch_size, num_workers=4, shuffle=(x == 'train'))
                   for x in phases}
    return datasets, dataloaders, 13 #datasets['train'].num_classes


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a point cloud classification network using 1D convs and hilbert order.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, default='/home/wangh0j/data/sfc/S3DIS_multiorder',
                        help='root directory containing S3DIS data')
    args = parser.parse_args()

    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    root_dir = args.root_dir
    phases = ['train', 'test']
    batch_size = 32
    category = 5

    # verify hilbert order.
    p = 7
    N = 2
    hilbert_curve = HilbertCurve(p, N)
    for coords in [[0, 0], [0, 1], [1, 1], [1, 0]]:
        dist = hilbert_curve.distance_from_coordinates(coords)
        print(f'distance(x={coords}) = {dist}')
    #
    datasets, dataloaders, num_classes = get_s3dis_dataloaders(root_dir=root_dir,
                                                               phases=phases,
                                                               batch_size=batch_size,
                                                               category=category)

    for phase in phases:
        print(phase.upper())
        print('\tDataset {} Dataloder {}'.format(len(datasets[phase]), len(dataloaders[phase])))
        for i, data in enumerate(dataloaders[phase]):  #what's the meaning?
            print(data.pos.shape)
            x = torch.cat((data.pos, data.x), dim=3).reshape(batch_size, 9, -1, 4) #zheli yao gaiyixia  data.pos = 4 * 4096 * 3 suoyi dim = 1
            seg_label = data.y
            print('\tData {} Seg Label {}'.format(x.size(), seg_label.size()))
            if i >= 3:
                break

    print('Test')
    print('\tDataset {} Dataloder {} Num classes {}'.format(len(datasets['test']),
                                                            len(dataloaders['test']), num_classes))
