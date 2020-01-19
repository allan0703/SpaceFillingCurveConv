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
from hilbertcurve.hilbertcurve import HilbertCurve
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob


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
        self.level_folder = 'level_'+str(self.level)
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
                    f = h5py.File(path)
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

                    if self.hilbert_order:
                        # order points in hilbert order
                        # after normalizaiton, -1 to 1
                        pos, x, y, norm = data.pos, data.x, data.y, data.norm
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
                        x = x[idx, :]
                        norm = norm[idx, :]
                        y = y[idx]
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


def get_s3dis_dataloaders(root_dir, phases, batch_size, dataset='sem_seg_h5', category='Bed', level=3, augment=False):
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
    datasets, dataloaders, num_classes = get_s3dis_dataloaders(root_dir=root_dir,
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
