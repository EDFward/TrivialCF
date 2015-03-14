import pickle
import os.path

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances

from utility import read_data_matrix


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


def k_nearest_neighbors(k, data_matrix, rows, similarity_measure, n_jobs=1):
    """
    Find k nearest neighbors for given rows of data_matrix. Used in
    memory-based CF.
    :param k: number of nearest users
    :param data_matrix: represents data points
    :param rows: rows to query of this data_matrix
    :param similarity_measure: similarity_measure: 'cosine' or 'dot_product'
    :return: nearest neighbors and corresponding similarities
    """
    nearest_neighbors = []
    # find k nearest neighbor for each user
    if similarity_measure == 'cosine':
        dist = pairwise_distances(data_matrix,
                                  data_matrix[rows, :],
                                  metric='cosine',
                                  n_jobs=n_jobs)
        similarities = 1 - dist
        # first count itself then remove it
        min_index = np.argpartition(dist, k + 1, axis=0)[:k + 1, :]
        # sort by distance for each column
        for i in range(min_index.shape[1]):
            col = min_index[:, i]
            col = col[np.argsort(dist[:, i][col])]
            # remove the most similar one: itself
            nearest_neighbors.append(col[1:])
    elif similarity_measure == 'dot_product':
        similarities = (data_matrix * data_matrix[rows, :].T).toarray()
        # make it column-wise distances matrix as above
        dist = -similarities
        min_index = np.argpartition(dist, k + 1, axis=0)[:k + 1, :]
        for i in range(min_index.shape[1]):
            col = min_index[:, i]
            col = col[np.argsort(dist[:, i][col])]
            # exclude the user itself
            if rows[i] in col:
                nn = np.delete(col, np.where(col == rows[i]))
            else:
                nn = col[:-1]
            nearest_neighbors.append(nn)
    return nearest_neighbors, similarities


def memory_cf(users, movies, k, similarity_measure, weight_schema):
    """
    Memory-based collaborative filtering.
    :param users: a user list. convert to a list if it's a single user
    :param movies: a movie list. convert to a list if it's a single movie
    :param k: number of nearest users
    :param similarity_measure: 'cosine' or 'dot_product'
    :param weight_schema: 'mean' or 'weighted_sum'
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

    nearest_neighbors, similarities = k_nearest_neighbors(k, rating_matrix,
                                                          users,
                                                          similarity_measure)
    ratings = []
    if weight_schema == 'mean':
        for neighbors, movie in zip(nearest_neighbors, movies):
            ratings.append(rating_matrix[neighbors, movie].sum() / k + 3)
    elif weight_schema == 'weighted_sum':
        for neighbors, movie, similarity in zip(nearest_neighbors, movies,
                                                similarities.T):
            neighbor_ratings = rating_matrix[neighbors, movie].toarray()
            rating = np.dot(neighbor_ratings.reshape(-1),
                            similarity[neighbors])
            rating /= similarity[neighbors].sum()
            rating += 3
            ratings.append(rating)

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
        neighbor_similarities = movie_cosine[movies, :].toarray()
    elif similarity_measure == 'dot_product':
        neighbor_similarities = movie_product[movies, :].toarray()

    dist = -neighbor_similarities
    min_index = np.argpartition(dist, k + 1, axis=1)[:, :k + 1]

    nearest_neighbors = []
    for i in range(min_index.shape[0]):
        row = min_index[i, :]
        row = row[np.argsort(dist[i, :][row])]
        # exclude the movie itself
        if movies[i] in row:
            nn = np.delete(row, np.where(row == movies[i]))
        else:
            nn = row[:-1]
        nearest_neighbors.append(nn)

    ratings = []
    if weight_schema == 'mean':
        for neighbors, user in zip(nearest_neighbors, users):
            ratings.append(rating_matrix[user, neighbors].sum() / k + 3)
    elif weight_schema == 'weighted_sum':
        for neighbors, user, similarity in zip(nearest_neighbors, users,
                                               neighbor_similarities):
            neighbor_ratings = rating_matrix[user, neighbors].toarray()
            rating = np.dot(neighbor_ratings.reshape(-1),
                            similarity[neighbors])
            rating /= similarity[neighbors].sum()
            rating += 3
            ratings.append(rating)

    return ratings


if __name__ == '__main__':
    pass
