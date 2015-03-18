import os
import pickle

from scipy.sparse import *
import numpy as np


def read_data_matrix(matrix_file_path='../train.csv'):
    with open(matrix_file_path) as f:
        entries = [map(int, line.split(',')[:-1])
                   for line in f.readlines()]

    movie_num = max(pair[0] for pair in entries) + 1
    user_num = max(pair[1] for pair in entries) + 1
    dict_matrix = dok_matrix((user_num, movie_num), dtype=np.float64)
    entry_set = set()

    for movie, user, rating in entries:
        # imputation by subtracting 3
        dict_matrix[user, movie] = rating - 3
        # recording entries
        entry_set.add((user, movie))

    if not os.path.isfile('../entry_set.pkl'):
        pickle.dump(entry_set, file('../entry_set.pkl', 'wb'))

    return dict_matrix.tocsr()


if __name__ == '__main__':
    data = read_data_matrix()
    pickle.dump(data, file('../train.pkl', 'wb'))
