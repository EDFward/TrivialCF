import pickle
import os.path
from itertools import ifilter, islice

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances
from sklearn import preprocessing
import sys
import time

from cluster import reinforce_cluster

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
    rating_matrix_orig = read_data_matrix()
else:
    rating_matrix_orig = get_data_matrix('../train.pkl', read_data_matrix)

# necessary data loading
entry_set = pickle.load(file('../entry_set.pkl'))


# two weighting methods
def mean(rating, similarity):
    return rating.sum() / max(rating.shape) + 3


def weighted_mean(rating, similarity):
    similarity -= similarity.min()
    sim_sum = similarity.sum()
    if sim_sum == 0:
        return 3
    else:
        similarity /= sim_sum
        return rating.dot(similarity) + 3


def memory_cf(users, movies, k, similarity_measure, weight_schema,
              repr_matrix=rating_matrix_orig, rating_matrix=rating_matrix_orig):
    """
    Memory-based collaborative filtering.
    :param users: a user list.
    :param movies: a movie list.
    :param k: number of nearest users
    :param similarity_measure: 'cosine' or 'dot_product'
    :param weight_schema: 'mean' or 'weighted_mean'
    :param repr_matrix: data point representation
    :param rating_matrix: ratings based on user-movie or cluster centroids
    :return: recommended ratings for the queries
    """

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
    rating_matrix = rating_matrix.toarray()
    weight_method = mean if weight_schema == 'mean' else weighted_mean

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

        rating = weight_method(rating_matrix[neighbors, movie],
                               sims[neighbor_index, neighbors])
        ratings.append(rating)

    return ratings


def model_cf(users, movies, k, similarity_measure, weight_schema,
             rating_matrix=rating_matrix_orig):
    """
    Model-based collaborative filtering. Same parameters as memory_cf
    except repr_matrix.
    """

    if similarity_measure == 'cosine':
        m2m = get_data_matrix('../movie-cosine.pkl',
                              lambda: sparse.csr_matrix(
                                  1 - pairwise_distances(
                                      rating_matrix_orig.T,
                                      metric='cosine')))
    elif similarity_measure == 'dot_product':
        m2m = get_data_matrix('../movie-dot.pkl',
                              lambda: rating_matrix_orig.T * rating_matrix_orig)

    # construct mapping between input movies and unique movies
    ratings, movie_unique = [], list(set(movies))
    movie_index_map = dict((m, i) for i, m in enumerate(movie_unique))
    movies = [(m, movie_index_map[m]) for m in movies]

    sims = m2m[movie_unique, :].toarray()
    dist = -sims

    # make rating matrix dense for fast access
    rating_matrix = rating_matrix.toarray()

    sorted_neighbors = np.argsort(dist, axis=1)
    weight_method = mean if weight_schema == 'mean' else weighted_mean

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

        rating = weight_method(rating_matrix[user, neighbors],
                               sims[neighbor_index, neighbors])
        ratings.append(rating)

    return ratings


def standardization_cf(users, movies, k, similarity_measure, weight_schema):
    """
    Memory-based collaborative filtering after standardization.
    Same parameters as model, with `similarity_measure` unused.
    """
    scaled_rating_matrix = preprocessing.scale(rating_matrix_orig.todense(),
                                               axis=1)
    return memory_cf(users, movies, k, 'dot_product', weight_schema,
                     repr_matrix=scaled_rating_matrix)


def cluster_cf(users, movies, k, similarity_measure, weight_schema,
               cluster_cf_type, k_user, k_movie):
    """
    Collaborative filtering using bipartite clustering, additional parameters
    are needed.
    :param cluster_cf_type: 'memory' or 'model'
    :param k_user: number of user clusters
    :param k_movie: number of movie clusters
    :return: recommended ratings for the queries
    """
    # record clustering time consumption
    start_time = time.time()
    cluster = reinforce_cluster(rating_matrix_orig, k_user, k_movie,
                                random_seed=False, iter_num=5)
    cluster_end_time = time.time()
    print 'cluster used %f seconds' % (cluster_end_time - start_time)
    (u2mc, user_belonging), (m2uc, movie_belonging) = cluster

    def cluster_cf_memory():
        """
        Cluster-based memory CF.
        """
        rating_matrix_cluster = np.empty([k_user, rating_matrix_orig.shape[1]],
                                         dtype=np.float64)

        # build rating matrix for each user cluster, on each movie
        for i in range(k_user):
            cluster_indicator = np.where(user_belonging == i)[0]
            rating_cluster = rating_matrix_orig[cluster_indicator, :]
            rating_sum = rating_cluster.sum(axis=0)
            # take average by dividing count
            rating_cluster.data = np.ones(len(rating_cluster.data))
            mu = rating_sum / rating_cluster.sum(axis=0)
            # fill 0 for nan
            mu[np.isnan(mu)] = 0
            rating_matrix_cluster[i, :] = mu

        # construct mapping between input users and unique users
        ratings, user_unique = [], list(set(users))
        user_index_map = dict((u, i) for i, u in enumerate(user_unique))
        users_neighbors = [user_index_map[u] for u in users]

        if similarity_measure == 'cosine':
            dist = cosine_distances(rating_matrix_orig[user_unique, :], m2uc.T)
            sims = 1 - dist
        else:
            sims = rating_matrix_orig[user_unique, :].dot(m2uc).toarray()
            dist = -sims

        nearest_neighbors = np.argpartition(dist, k, axis=1)[:, :k]
        weight_method = mean if weight_schema == 'mean' else weighted_mean

        for neighbor_index, movie in zip(users_neighbors, movies):
            neighbors = nearest_neighbors[neighbor_index]
            rating = weight_method(rating_matrix_cluster[neighbors, movie],
                                   sims[neighbor_index, neighbors])
            ratings.append(rating)

        return ratings

    def cluster_cf_model():
        """
        Cluster-based model CF.
        """
        rating_matrix_cluster = np.empty([rating_matrix_orig.shape[0], k_movie],
                                         dtype=np.float64)
        # build rating matrix for each user, on each movie cluster
        for i in range(k_movie):
            cluster_indicator = np.where(movie_belonging == i)[0]
            rating_cluster = rating_matrix_orig[:, cluster_indicator]
            rating_sum = rating_cluster.sum(axis=1)
            # divide by count for average
            rating_cluster.data = np.ones(len(rating_cluster.data))
            mu = rating_sum / rating_cluster.sum(axis=1)
            # fill 0 for nan
            mu[np.isnan(mu)] = 0
            mu = mu.A.squeeze()
            rating_matrix_cluster[:, i] = mu

        # construct mapping between input movies and unique movies
        ratings, movie_unique = [], list(set(movies))
        movie_index_map = dict((m, i) for i, m in enumerate(movie_unique))
        movie_neighbors = [movie_index_map[m] for m in movies]

        # |m| by k_m matrix of movie representation
        m2mc = rating_matrix_orig.T * u2mc

        sims = m2mc[movie_unique, :].toarray()
        dist = -sims
        nearest_neighbors = np.argpartition(dist, k, axis=1)[:, :k]
        weight_method = mean if weight_schema == 'mean' else weighted_mean

        for user, neighbor_index in zip(users, movie_neighbors):
            neighbors = nearest_neighbors[neighbor_index]
            rating = rating_matrix_cluster[user, neighbors].sum() / k + 3
            rating = weight_method(rating_matrix_cluster[user, neighbors],
                                   sims[neighbor_index, neighbors])
            ratings.append(rating)

        return ratings

    if cluster_cf_type == 'memory':
        ratings = cluster_cf_memory()
    elif cluster_cf_type == 'model':
        ratings = cluster_cf_model()
    else:
        print '==ERROR== wrong arguments for cluster CF'
        sys.exit(1)

    print 'cf used %f seconds' % (time.time() - cluster_end_time)
    return ratings
