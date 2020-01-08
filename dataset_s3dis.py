import numpy as np
import os
from tqdm import tqdm, trange
import glob
import h5py
import logging
from torch.utils.data import Dataset, DataLoader
# from sklearn.neighbors import KDTree

# append upper directory to import point_order.py
import utils as utl
from hilbertcurve.hilbertcurve import HilbertCurve


def load_data(root_dir, test_area):
    """
    Load S3DIS data from h5 files

    :param root_dir: Directory with shapenet h5 data
    :param test_area: Area used for test set (1, 2, 3, 4, 5, or 6)

    :return: 2 Numpy arrays for data (PxNx9), segmentation label (PxN)
    """
    # first we load the room file list
    with open(os.path.join(root_dir, 'room_filelist.txt'), 'r') as f:
        room_list = np.array([line.rstrip() for line in f.readlines()])

    # load filenames
    with open(os.path.join(root_dir, 'all_files.txt'), 'r') as f:
        all_files = np.array([os.path.join(root_dir, os.path.basename(line.rstrip())) for line in f.readlines()])

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

    # k = 9
    # adj = np.zeros((all_data.shape[0], all_data.shape[1], k), dtype=np.int)
    # for i in trange(all_data.shape[0], desc='Computing neighbors'):
    #     tree = KDTree(all_data[i, :, :3], leaf_size=50, metric='euclidean')
    #     adj[i, ...] = tree.query(all_data[i, :, :3], k=k, return_distance=False)

    logging.info('Computing number of classes...')
    num_classes = np.unique(all_label).shape[0]

    logging.info('Creating data splits for Area {}'.format(test_area))
    # now we split data into train or test based on test area
    test_idx = ['Area_{}'.format(test_area) in x for x in room_list]
    train_idx = ['Area_{}'.format(test_area) not in x for x in room_list]

    train_data = all_data[train_idx, ...]
    train_label = all_label[train_idx, ...]
    # train_adj = adj[train_idx, ...]

    test_data = all_data[test_idx, ...]
    test_label = all_label[test_idx, ...]
    # test_adj = adj[test_idx, ...]

    return train_data, train_label, test_data, test_label, num_classes
    # return train_data, train_label, test_data, test_label, train_adj, test_adj, num_classes


class S3DIS(Dataset):
    def __init__(self, data_label, augment=False):
        self.augment = augment
        self.data, self.label = data_label
        self.p = 7
        # compute hilbert order for voxelized space
        logging.info('Computing hilbert distances...')
        self.hilbert_curve = HilbertCurve(self.p, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        pointcloud = self.data[item, ...]
        label = self.label[item, ...]
        # nns = self.adj[item, ...]

        if self.augment:
            pointcloud = utl.augment_point_cloud(pointcloud)

        # normalize points
        # todo: do we have to do normalization?
        points_norm = pointcloud - pointcloud.min(axis=0)
        points_norm /= points_norm.max(axis=0) + 1e-23

        # order points in hilbert order
        points_voxel = np.floor(points_norm * (2 ** self.p - 1))
        hilbert_dist = np.zeros(points_voxel.shape[0])

        # todo: we want to try two methods.
        # todo: 1. globally along with locally hilbert curve
        # todo: 2. multi-level of hilbert curve

        for i in range(points_voxel.shape[0]):
            # by doing the astype int, it will assign the same hilbert_dist to the points that belong to the same space
            # todo: check how much duplicates (multi-points in the same space).
            #  answer is no. p is 7, which partition the space very precise
            hilbert_dist[i] = self.hilbert_curve.distance_from_coordinates(points_voxel[i, 0:3].astype(int))
        idx = np.argsort(hilbert_dist)

        return pointcloud[idx, :], label


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
    # first load data
    train_data, train_label, test_data, test_label, num_classes = load_data(root_dir=root_dir,
                                                                                                 test_area=category)

    datasets = {
        'train': S3DIS(data_label=(train_data, train_label), augment=augment),
        'test': S3DIS(data_label=(test_data, test_label))
    }

    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, num_workers=4, shuffle=(x == 'train'))
                   for x in phases}

    return datasets, dataloaders, num_classes


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    root_dir = '/data/sfc/indoor3d_sem_seg_hdf5_data'
    phases = ['train', 'test']
    batch_size = 32
    category = 5

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
