import numpy as np
import os
from tqdm import tqdm, trange
import glob
import h5py
import logging

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
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()

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
    trans = 0.05 * np.random.randn(1, 3)

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
    rotation_angle = np.random.uniform() * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    rotated_data = np.dot(pointcloud, rotation_matrix)

    return rotated_data.astype('float32')


def scale_pointcloud(pointcloud):
    """ Randomly scale a pointcloud
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, scaled batch of point clouds
    """
    scale = 0.1 * np.random.rand(3) + 1.0
    scale = np.clip(scale, 0.7, 1.3)
    points_scaled = pointcloud * scale

    return points_scaled.astype('float32')


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


class ModelNet40(Dataset):
    def __init__(self, root_dir, phase='train', augment=False):
        self.phase = phase
        self.augment = augment
        self.data, self.label = load_data(root_dir=root_dir, phase=phase)
        self.p = 7

        # compute hilbert order for voxelized space
        logging.info('Computing hilbert distances...')
        self.hilbert_curve = HilbertCurve(self.p, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        pointcloud = self.data[item, ...]
        label = self.label[item]

        # augment data when requested
        if self.augment:
            pointcloud = rotate_pointcloud(pointcloud)
            # pointcloud = translate_pointcloud(pointcloud)
            pointcloud = scale_pointcloud(pointcloud)
            # pointcloud = shear_pointcloud(pointcloud)

        # normalize points
        points_norm = pointcloud - pointcloud.min(axis=0)
        points_norm /= points_norm.max(axis=0) + 1e-23

        # order points in hilbert order
        points_voxel = np.floor(points_norm * (2 ** self.p - 1))
        hilbert_dist = np.zeros(points_voxel.shape[0])
        for i in range(points_voxel.shape[0]):
            hilbert_dist[i] = self.hilbert_curve.distance_from_coordinates(points_voxel[i, :].astype(int))
        idx = np.argsort(hilbert_dist)

        return pointcloud[idx, :], label


def get_modelnet40_dataloaders(root_dir, phases, batch_size, augment=True):
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
    datasets = {x: ModelNet40(root_dir=root_dir, phase=x, augment=(x == 'train') and augment) for x in phases}

    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, num_workers=4, shuffle=(x == 'train'))
                   for x in phases}

    return datasets, dataloaders


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    root_dir = '/media/thabetak/a5411846-373b-430e-99ac-01222eae60fd/ModelNet/modelnet40_ply_hdf5_2048'
    phases = ['train', 'test']
    batch_size = 32
    augment = False

    datasets, dataloaders = get_modelnet40_dataloaders(root_dir, phases, batch_size, augment)

    for phase in phases:
        print(phase.upper())
        print('\tDataset {} Dataloder {}'.format(len(datasets[phase]), len(dataloaders[phase])))
        for i, (data, label) in enumerate(dataloaders[phase]):
            print('\tData {} Label {}'.format(data.size(), label.size()))
            if i >= 3:
                break

    print('Test')
    print('\tDataset {} Dataloder {}'.format(len(datasets['test']), len(dataloaders['test'])))
