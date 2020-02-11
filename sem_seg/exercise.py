import numpy as np


def get_edge_index_index(num_points, num_neighbors):
    edge_index_index = np.zeros((num_points, num_neighbors))
    for i in range(num_points):
        if num_neighbors / 2 <= i < (num_points - 5):
            edge_index_index[i] = np.concatenate((np.arange(i - 4, i), np.arange(i + 1, i + 6)))
        elif i < num_neighbors / 2:
            l = np.arange(num_neighbors + 1)
            l = filter(lambda x: x != i, l)
            l = [j for j in l]
            edge_index_index[i] = l
        else:
            l = np.arange(num_points - 10, num_points)
            l = filter(lambda x: x != i, l)
            l = [j for j in l]
            edge_index_index[i] = l

    return edge_index_index

a = get_edge_index_index(50,9)
print(a.shape,a)