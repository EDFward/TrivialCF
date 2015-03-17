import pickle
import os.path

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing

from utility import read_data_matrix
from itertools import izip


def get_data_matrix(pickle_file_path, generate_method):
    if os.path.isfile(pickle_file_path):
        return pickle.load(file(pickle_file_path))
    else:
        data = generate_method()
        pickle.dump(data, file(pickle_file_path, 'wb'))
        return data


rating_matrix = get_data_matrix('../train.pkl', read_data_matrix)
movie_product = get_data_matrix('../movie-dot.pkl',
                                lambda: rating_matrix.T * rating_matrix)
movie_cosine = get_data_matrix('../movie-cosine.pkl',
                               lambda: sparse.csr_matrix(
                                   1 - pairwise_distances(rating_matrix.T,
                                                          metric='cosine')))


def memory_cf(users, movies, k, similarity_measure, weight_schema,
              data_matrix=rating_matrix):
    """
    Memory-based collaborative filtering.
    :param users: a user list. convert to a list if it's a single user
    :param movies: a movie list. convert to a list if it's a single movie
    :param k: number of nearest users
    :param similarity_measure: 'cosine' or 'dot_product'
    :param weight_schema: 'mean' or 'weighted_sum'
    :param data_matrix: user ratings
    :return: recommended ratings for the queries
    """
    # argument sanity check
    if (similarity_measure not in ('cosine', 'dot_product') or
            weight_schema not in ('mean', 'weighted_sum')):
        print '==ERROR== unsupported arguments for memory CF'
        return None
    if type(users) != list:
        users = [users]
    if type(movies) != list:
        movies = [movies]

    # find k nearest neighbor for each user
    if similarity_measure == 'cosine':
        dist = pairwise_distances(data_matrix[users, :],
                                  data_matrix,
                                  metric='cosine')
        similarities = 1 - dist
    elif similarity_measure == 'dot_product':
        similarities = (data_matrix[users, :] * data_matrix.T).toarray()
        dist = -similarities

    # first count itself then remove it
    min_index = np.argpartition(dist, k + 1, axis=1)[:, :k + 1]

    ratings = []
    if weight_schema == 'mean':
        for i, min_row in enumerate(min_index):
            neighbors = np.delete(min_row, np.where(min_row == users[i]))
            ratings.append(data_matrix[neighbors, movies[i]].sum() / k + 3)
    elif weight_schema == 'weighted_sum':
        for i, (min_row, sim) in enumerate(izip(min_index, similarities)):
            neighbors = np.delete(min_row, np.where(min_row == users[i]))
            neighbor_ratings = data_matrix[neighbors, movies[i]]
            rating = neighbor_ratings.T.dot(sim[neighbors]).squeeze()
            rating /= sim[neighbors].sum()
            rating += 3
            ratings.append(rating if not np.isnan(rating) else 3)

    return ratings


def model_cf(users, movies, k, similarity_measure, weight_schema):
    """
    Model-based collaborative filtering. Same parameters as memory_cf.
    """
    # argument sanity check
    if (similarity_measure not in ('cosine', 'dot_product') or
            weight_schema not in ('mean', 'weighted_sum')):
        print '==ERROR== unsupported arguments for model CF'
        return None
    if type(users) != list:
        users = [users]
    if type(movies) != list:
        movies = [movies]

    if similarity_measure == 'cosine':
        similarities = movie_cosine[movies, :].toarray()
    elif similarity_measure == 'dot_product':
        similarities = movie_product[movies, :].toarray()

    dist = -similarities
    min_index = np.argpartition(dist, k + 1, axis=1)[:, :k + 1]

    ratings = []
    if weight_schema == 'mean':
        for i, min_row in enumerate(min_index):
            neighbors = np.delete(min_row, np.where(min_row == movies[i]))
            ratings.append(rating_matrix[users[i], neighbors].sum() / k + 3)
    elif weight_schema == 'weighted_sum':
        for i, (min_row, sim) in enumerate(izip(min_index, similarities)):
            neighbors = np.delete(min_row, np.where(min_row == movies[i]))
            neighbor_ratings = rating_matrix[users[i], neighbors]
            rating = neighbor_ratings.dot(sim[neighbors]).squeeze()
            rating /= sim[neighbors].sum()
            rating += 3
            ratings.append(rating if not np.isnan(rating) else 3)

    return ratings


def standardization_cf(users, movies, k, weight_schema):
    return memory_cf(users, movies, k, 'dot_product', weight_schema,
                     data_matrix=preprocessing.scaled(rating_matrix))


if __name__ == '__main__':
    pass
