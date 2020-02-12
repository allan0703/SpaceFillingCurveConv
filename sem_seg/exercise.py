import numpy as np


# def get_edge_index_index(num_points, num_neighbors):
#     edge_index_index = np.zeros((num_points, num_neighbors))
#     for i in range(num_points):
#         if num_neighbors / 2 <= i < (num_points - 5):
#             edge_index_index[i] = np.concatenate((np.arange(i - 4, i), np.arange(i + 1, i + 6)))
#         elif i < num_neighbors / 2:
#             l = np.arange(num_neighbors + 1)
#             l = filter(lambda x: x != i, l)
#             l = [j for j in l]
#             edge_index_index[i] = l
#         else:
#             l = np.arange(num_points - 10, num_points)
#             l = filter(lambda x: x != i, l)
#             l = [j for j in l]
#             edge_index_index[i] = l
#     center_point = np.arange(num_points).transpose()
#     center_point = np.repeat(center_point[:, np.newaxis], num_neighbors, axis=1)
#     edge_index = np.stack((idx[center_point.astype(np.int16)], idx[edge_index_index.astype(np.int16)]), axis=0)
#
#     a = np.random.randint(0, num_points, size=(num_points, 1))
#     b = a[edge_index_index.astype(np.int16)]
#     b = np.squeeze(b, axis= 2)
#
#     return edge_index_index
def get_edge_index_index(idx, k=9):
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
    edge_index = np.squeeze(edge_index, axis=3)
    return edge_index

a = np.random.randint(0, 50, size= (50,1))

b = get_edge_index_index(a, 9)
print(b.shape, b)