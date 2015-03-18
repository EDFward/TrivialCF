import pickle
import os.path
from itertools import ifilter, islice

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances
from sklearn import preprocessing

from utility import read_data_matrix


def get_data_matrix(pickle_file_path, generate_method):
    if os.path.isfile(pickle_file_path):
        return pickle.load(file(pickle_file_path))
    else:
        data = generate_method()
        pickle.dump(data, file(pickle_file_path, 'wb'))
        return data

# make sure the entries exist, otherwise regenerate
if not os.path.isfile('../entry_set.pkl'):
    rating_matrix = read_data_matrix()
else:
    rating_matrix = get_data_matrix('../train.pkl', read_data_matrix)

# necessary data loading
entry_set = pickle.load(file('../entry_set.pkl'))


def memory_cf(users, movies, k, similarity_measure, weight_schema,
              repr_matrix=rating_matrix):
    """
    Memory-based collaborative filtering.
    :param users: a user list. convert to a list if it's a single user
    :param movies: a movie list. convert to a list if it's a single movie
    :param k: number of nearest users
    :param similarity_measure: 'cosine' or 'dot_product'
    :param weight_schema: 'mean' or 'weighted_mean'
    :param repr_matrix: data point representation
    :return: recommended ratings for the queries
    """
    # argument sanity check
    if (similarity_measure not in ('cosine', 'dot_product') or
                weight_schema not in ('mean', 'weighted_mean')):
        print '==ERROR== unsupported arguments for memory CF'
        return None
    if type(users) != list:
        users = [users]
    if type(movies) != list:
        movies = [movies]

    # construct mapping between input users and unique users
    ratings, user_unique = [], list(set(users))
    user_index_map = dict((u, i) for i, u in enumerate(user_unique))
    users = [(u, user_index_map[u]) for u in users]

    # find k nearest neighbor for each user
    if similarity_measure == 'cosine':
        dist = cosine_distances(repr_matrix[user_unique, :], repr_matrix)
        sims = 1 - dist
    elif similarity_measure == 'dot_product':
        sims = repr_matrix[user_unique, :].dot(repr_matrix.T)
        if issparse(sims):
            sims = sims.toarray()
        dist = -sims

    sorted_neighbors = np.argsort(dist, axis=1)

    # make rating matrix dense for fast access
    rating_matrix_dense = rating_matrix.todense()

    for (user_index, neighbor_index), movie in zip(users, movies):
        neighbors = list(islice(ifilter(lambda u: (u, movie) in entry_set,
                                        sorted_neighbors[neighbor_index]),
                                k + 1))

        # no neighbors, regarded as 3
        if not neighbors:
            ratings.append(3)
            continue

        # exclude itself
        if user_index in neighbors:
            neighbors.remove(user_index)

        if weight_schema == 'mean':
            rating = rating_matrix_dense[neighbors, movie].sum() / len(
                neighbors) + 3
        else:
            sim = sims[neighbor_index, neighbors]
            sim -= sim.min()
            sim_sum = sim.sum()
            if sim_sum == 0:
                rating = 3
            else:
                sim /= sim_sum
                rating = rating_matrix_dense[neighbors, movie].T.dot(
                    sim).A.squeeze() + 3
        ratings.append(rating)

    return ratings


def model_cf(users, movies, k, similarity_measure, weight_schema):
    """
    Model-based collaborative filtering. Same parameters as memory_cf
    except repr_matrix.
    """
    # argument sanity check
    if (similarity_measure not in ('cosine', 'dot_product') or
                weight_schema not in ('mean', 'weighted_mean')):
        print '==ERROR== unsupported arguments for model CF'
        return None
    if type(users) != list:
        users = [users]
    if type(movies) != list:
        movies = [movies]

    if similarity_measure == 'cosine':
        m2m = get_data_matrix('../movie-cosine.pkl',
                              lambda: sparse.csr_matrix(
                                  1 - pairwise_distances(
                                      rating_matrix.T,
                                      metric='cosine')))
    elif similarity_measure == 'dot_product':
        m2m = get_data_matrix('../movie-dot.pkl',
                              lambda: rating_matrix.T * rating_matrix)

    # construct mapping between input movies and unique movies
    ratings, movie_unique = [], list(set(movies))
    movie_index_map = dict((m, i) for i, m in enumerate(movie_unique))
    movies = [(m, movie_index_map[m]) for m in movies]

    sims = m2m[movie_unique, :].toarray()
    dist = -sims

    # make rating matrix dense for fast access
    rating_matrix_dense = rating_matrix.todense()

    sorted_neighbors = np.argsort(dist, axis=1)

    for user, (movie_index, neighbor_index) in zip(users, movies):
        neighbors = list(islice(ifilter(lambda m: (user, m) in entry_set,
                                        sorted_neighbors[neighbor_index]),
                                k + 1))

        # no neighbors, regarded as 3
        if not neighbors:
            ratings.append(3)
            continue

        # exclude itself
        if movie_index in neighbors:
            neighbors.remove(movie_index)

        if weight_schema == 'mean':
            rating = rating_matrix_dense[user, neighbors].sum() / len(
                neighbors) + 3
        else:
            sim = sims[neighbor_index, neighbors]
            sim -= sim.min()
            sim_sum = sim.sum()
            if sim_sum == 0:
                rating = 3
            else:
                sim /= sim_sum
                rating = rating_matrix_dense[user, neighbors].dot(
                    sim).A.squeeze() + 3
        ratings.append(rating)

    return ratings


def standardization_cf(users, movies, k, similarity_measure, weight_schema):
    """
    Memory-based collaborative filtering after standardization.
    Same parameters as model, with `similarity_measure` unused.
    """
    scaled_rating_matrix = preprocessing.scale(rating_matrix.todense(), axis=1)
    return memory_cf(users, movies, k, 'dot_product', weight_schema,
                     repr_matrix=scaled_rating_matrix)


if __name__ == '__main__':
    pass
