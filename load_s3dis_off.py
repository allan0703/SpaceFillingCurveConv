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
            self.hilbert_curve = HilbertCurve(p, 3)
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
            xs += torch.from_numpy(f['data'][:]).unbind(0)
            ys += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)

        test_area = 'Area_{}'.format(self.test_area)
        train_data_list, test_data_list = [], []
        for i, (x, y) in enumerate(tqdm(zip(xs, ys), total=len(xs))):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Hilbert curve
            if self.hilbert_order:
                # order points in hilbert order
                # after normalizaiton, -1 to 1
                pos, x, y = data.pos, data.x, data.y
                point = (pos.numpy()+1)/2  # normalize to 0, 1
                points_voxel = np.floor(point * (2 ** self.p - 1)).astype(int)
                hilbert_dist = np.zeros(points_voxel.shape[0])

                # todo: we want to try two methods.
                # todo: 1. globally along with locally hilbert curve
                # todo: 2. multi-level of hilbert curve
                for point_idx in range(points_voxel.shape[0]):
                    hilbert_dist[point_idx] = self.hilbert_curve.distance_from_coordinates(points_voxel[point_idx, :])
                idx = np.argsort(hilbert_dist)
                pos = pos[idx, :]
                x = x[idx, :]
                y = y[idx]
                data = Data(pos=pos, x=x, y=y)

            if test_area not in rooms[i]:
                train_data_list.append(data)
            else:
                test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])


def get_s3dis_dataloaders(root_dir, phases, batch_size, category, augment=False):
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
        'train': S3DIS(root_dir, 5, True, pre_transform=T.NormalizeScale()),
        'test': S3DIS(root_dir, 5, False, pre_transform=T.NormalizeScale())
    }

    dataloaders = {x: DenseDataLoader(datasets[x], batch_size=batch_size, num_workers=4, shuffle=(x == 'train'))
                   for x in phases}
    return datasets, dataloaders, num_classes


if __name__ == '__main__':

    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    root_dir = '/data/sfc/S3DIS'
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
        for i, (data, seg_label) in enumerate(dataloaders[phase]):
            print('\tData {} Seg Label {}'.format(data.size(), seg_label.size()))
            if i >= 3:
                break

    print('Test')
    print('\tDataset {} Dataloder {} Num classes {}'.format(len(datasets['test']),
                                                            len(dataloaders['test']), num_classes))
