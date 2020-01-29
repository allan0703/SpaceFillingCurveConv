import numpy as np
import os
import glob
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from model.deeplab import deeplab
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree
from S3DIS import S3DISDataset
from config import S3DISConfig


def sample_data(data, num_sample):
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), np.concatenate((np.arange(N), sample)).astype(int)


#     list(range(N)) + list(sample)


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]

    return new_data, new_label, sample_indices


def room2blocks(data, label, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    assert (stride <= block_size)

    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    block_idx_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
            continue

        block_data = data[cond, :]
        block_label = label[cond]

        cond_idxs = np.where(cond)[0]

        # randomly subsample data
        block_data_sampled, block_label_sampled, sample_indices = sample_data_label(block_data,
                                                                                    block_label,
                                                                                    num_point)
        #         print(cond_idxs)
        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))
        block_idx_list.append(np.expand_dims(cond_idxs[sample_indices], 0))

    return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0), np.concatenate(block_idx_list, 0)


def room2blocks_plus_normalized(data_label_filename, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    data_label = np.load(data_label_filename)

    data = data_label[:, 0:6]
    data[:, 3:6] /= 255.0
    label = data_label[:, -1].astype(np.uint8)
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    data_batch, label_batch, idxs = room2blocks(data, label, num_point, block_size, stride,
                                                random_sample, sample_num, sample_aug)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx + block_size / 2)
        data_batch[b, :, 1] -= (miny + block_size / 2)
    new_data_batch[:, :, 0:6] = data_batch

    return new_data_batch, label_batch, idxs


def get_block_data_for_area(root_dir, test_area, num_point=4096, block_size=1.0, stride=1.0,
                            random_sample=False, sample_num=None, sample_aug=1):
    room_data = {}
    # print(os.path.join(root_dir, 'Area_{:d}*.npy'.format(test_area)))
    # print(glob.glob(os.path.join(root_dir, 'Area_{:d}*.npy'.format(test_area))))
    for filename in tqdm(glob.glob(os.path.join(root_dir, 'Area_{:d}*.npy'.format(test_area))),
                         desc='Getting block data for Area {:d}'.format(test_area)):
        data, label, idxs = room2blocks_plus_normalized(filename, num_point, block_size, stride,
                                                        random_sample, sample_num, sample_aug)
        room_name = os.path.basename(filename).replace('.npy', '')
        room_data[room_name] = {
            'data': data,
            'label': label,
            'idx': idxs
        }
        break

    return room_data


def test_s3dis_segmentation(root_dir, model_path, config, k=5):
    room_data = get_block_data_for_area(root_dir, config.test_area)

    best_state_path = os.path.join(model_path, 'best_state.pth')
    best_state = torch.load(best_state_path)

    # device = torch.device('cuda:{}'.format(config.gpu_index))
    device = torch.device('cuda:0')
    model = deeplab(backbone=config.backbone, input_size=config.num_feats,
                    num_classes=config.num_classes, kernel_size=config.kernel_size,
                    sigma=config.sigma).to(device)
    model.load_state_dict(best_state['model'])
    model.eval()

    room_preds = {}
    for room, vals in tqdm(room_data.items(), desc='Predicting labels on all room'):
        all_outs = []
        all_labels = []
        all_points = []

        dataset = S3DISDataset((vals['data'], vals['label']), num_features=9)
        dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=4, shuffle=False)

        for inputs in tqdm(dataloader, desc='Evaluating on room {}'.format(room)):
            data = inputs[0]
            data = torch.cat((data[:, :, 3:6], data[:, :, 2].unsqueeze(-1)), dim=-1)
            data = data.to(device, dtype=torch.float).permute(0, 2, 1)
            coords = inputs[1].to(device, dtype=torch.float).permute(0, 2, 1)
            label = inputs[2]

            out = model(data, coords)

            all_points.append(inputs[0][:, :, 6:].numpy())
            all_outs.append(out.detach().cpu().numpy())
            all_labels.append(label)

            room_preds[room] = {
                'points': np.concatenate(all_points, 0),
                'logits': np.concatenate(all_outs, 0),
                'labels': np.concatenate(all_labels, 0)
            }

    cm_full = np.zeros((config.num_classes, config.num_classes))
    cm_blocks = np.zeros((config.num_classes, config.num_classes))
    for room, vals in tqdm(room_preds.items(), desc='Computing accuracy on full data'):
        data_label = np.load(os.path.join(root_dir, '{}.npy'.format(room)))
        points = vals['points']
        labels = vals['labels']
        logits = vals['logits'].transpose(0, 2, 1)
        preds = logits.argmax(axis=-1)

        maxs = data_label[:, :3].max(axis=0)
        points_unique, unique_idx = np.unique(points.reshape(-1, 3) * maxs, return_index=True, axis=0)

        preds_unique = preds.reshape(-1)[unique_idx]

        ball_tree = BallTree(points_unique, metric='euclidean')
        knn_classes = preds_unique[ball_tree.query(data_label[:, :3], k=k)[1]].astype(int)

        interpolated = np.zeros(knn_classes.shape[0])
        for i in trange(knn_classes.shape[0], desc='Interpolating labels'):
            interpolated[i] = np.bincount(knn_classes[i]).argmax()

        cm_full += confusion_matrix(data_label[:, -1], interpolated, labels=np.arange(config.num_classes))
        cm_blocks += confusion_matrix(labels.reshape(-1), preds.reshape(-1), labels=np.arange(config.num_classes))

    # Blocks accuracy
    acc = 100.0 * np.diag(cm_blocks).sum() / (cm_blocks.sum() + 1e-15)
    class_acc = 100.0 * np.diag(cm_blocks) / (cm_blocks.sum(axis=1) + 1e-15)
    iou = 100.0 * np.diag(cm_blocks) / (cm_blocks.sum(axis=1) + cm_blocks.sum(axis=0) - np.diag(cm_blocks) + 1e-15)

    print('--------------------------------------------------------------------------------')
    print('Blocks accuracy:')
    print('Overall Acc: {:.2f}. Class Acc: {:.2f}. mIoU {:.2f}'.format(acc, class_acc.mean(), iou.mean()))
    iou_per_class_str = ' '.join(['{:.2f}'.format(s) for s in iou])
    print('IoU per class: {}'.format(iou_per_class_str))
    print('--------------------------------------------------------------------------------')

    # Full accuracy
    acc = 100.0 * np.diag(cm_full).sum() / (cm_full.sum() + 1e-15)
    class_acc = 100.0 * np.diag(cm_full) / (cm_full.sum(axis=1) + 1e-15)
    iou = 100.0 * np.diag(cm_full) / (cm_full.sum(axis=1) + cm_full.sum(axis=0) - np.diag(cm_full) + 1e-15)

    print('--------------------------------------------------------------------------------')
    print('Full accuracy:')
    print('Overall Acc: {:.2f}. Class Acc: {:.2f}. mIoU {:.2f}'.format(acc, class_acc.mean(), iou.mean()))
    iou_per_class_str = ' '.join(['{:.2f}'.format(s) for s in iou])
    print('IoU per class: {}'.format(iou_per_class_str))
    print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
    root_dir = '/media/thabetak/a5411846-373b-430e-99ac-01222eae60fd/S3DIS/data_label_npy'
    model_path = '/media/thabetak/a5411846-373b-430e-99ac-01222eae60fd/S3DIS/checkpoints/deeplab/err_fixed_sigma/2020-01-23-16-12-17_S3DIS_5_4_0.001_8_15_0.15_deeplab_resnet18_augment_no-bias_300__109d2264-8a61-4b9c-b6ff-6356e06dd7dc'
    config = S3DISConfig()
    config.load(os.path.join(model_path, 'config.txt'))

    test_s3dis_segmentation(root_dir, model_path, config)


