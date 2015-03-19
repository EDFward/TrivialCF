import subprocess
from operator import itemgetter

import sys
from scipy.sparse import issparse, isspmatrix_csr
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine

from utility import *


def k_means(data_matrix, k, starting_points=None, random_seed=True,
            soft_assign=False):
    """
    K-Means cluster for data points.
    :param data_matrix: matrix with rows representing data points
    :param k: number of clusters
    :param starting_points: seed centroids
    :param random_seed: using randomly pickled centroids at start, or
            choose the least similar ones iteratively
    :param soft_assign: using soft assignment for cluster belonging
    :return: cluster centroids and belonging for every data point
    """
    row_num = data_matrix.shape[0]
    # transform to row-major matrix if possible
    if issparse(data_matrix) and not isspmatrix_csr(data_matrix):
        data_matrix = data_matrix.tocsr()

    if starting_points is None:
        if not random_seed:
            # gradually pick farthest point from already picked starting points
            starting_points = np.empty((k, data_matrix.shape[1]),
                                       dtype=np.float64)
            rand_index = np.random.randint(row_num)
            starting_points[0, :] = data_matrix[rand_index, :].toarray()

            nearest_dist = np.ones(row_num)
            for i in range(1, k):
                new_dist = pairwise_distances(data_matrix,
                                              starting_points[i - 1, :],
                                              metric='cosine',
                                              n_jobs=1).reshape(-1)
                nearest_dist = np.min([new_dist, nearest_dist], axis=0)
                farthest = max(enumerate(nearest_dist), key=itemgetter(1))[0]
                starting_points[i, :] = data_matrix[farthest, :].toarray()
        else:
            # randomly pick starting points
            random_index = np.random.choice(row_num, k, replace=False)
            starting_points = data_matrix[random_index, :].toarray()

    cluster_centers = starting_points
    cluster_belonging = np.empty(row_num)

    converged = False
    while not converged:
        distances = pairwise_distances(data_matrix, cluster_centers,
                                       metric='cosine', n_jobs=1)
        belonging = [min(enumerate(row), key=itemgetter(1))[0]
                     for row
                     in distances]
        belonging = np.asarray(belonging)

        new_centers = cluster_centers.copy()

        for i in range(k):
            cluster_indicator = np.where(belonging == i)[0]
            if not cluster_indicator.any():
                # no points belong to this cluster, randomly pick a point
                mu = data_matrix[np.random.randint(row_num), :].toarray()
            else:
                mu = data_matrix[cluster_indicator, :].mean(axis=0)
            new_centers[i, :] = mu

        # converge if centroids are not so far away
        converged = all(cosine(center_before, center_curr) < 0.05
                        for center_before, center_curr
                        in zip(cluster_centers, new_centers))

        cluster_centers = new_centers
        cluster_belonging = belonging

    # using soft assignment if specified
    if soft_assign:
        for i, j in enumerate(cluster_belonging):
            cluster_belonging[i] = 1 - distances[i, j]

    return cluster_centers, cluster_belonging


def cosine_sum(data_matrix, k, belonging):
    """
    Internal metric to sum cosine similarity between data points and
    corresponding clusters.
    :param data_matrix: matrix with rows representing data points
    :param k: number of clusters
    :param belonging: indications for point and corresponding clusters
    :return: sum of cosine similarity
    """
    centers = np.empty([k, data_matrix.shape[1]], dtype=np.float64)
    for i in range(k):
        cluster_indicator = np.where(belonging == i)[0]
        mu = data_matrix[cluster_indicator, :].mean(axis=0)
        centers[i, :] = mu
    distances = pairwise_distances(data_matrix, centers,
                                   metric='cosine', n_jobs=1)
    return sum(1 - distances[r, c] for r, c in enumerate(belonging))


def reinforce_cluster(doc_vectors, k_doc, k_word, random_seed, iter_num=5):
    """
    Iteratively cluster documents and words.
    :param doc_vectors: a sparse matrix for document vectors
    :param k_doc: number of document clusters
    :param k_word: number of word clusters
    :param iter_num: iteration number
    :param random_seed: use random initialization for k-means or not
    :return: document repr/belonging, word repr/belonging
    """
    # initial w2dc
    w2dc = doc_vectors.transpose()

    for i in range(iter_num):
        wc_center, wc_belonging = k_means(w2dc, k_word, random_seed=random_seed)
        # transform cluster_belonging to indicator matrix
        w2wc = dok_matrix((len(wc_belonging), k_word), dtype=np.float64)
        for w, wc in enumerate(wc_belonging):
            w2wc.update({(w, wc): 1})
        w2wc = w2wc.tocsc()

        d2wc = doc_vectors * w2wc
        dc_center, dc_belonging = k_means(d2wc, k_doc, random_seed=random_seed)
        d2dc = dok_matrix((len(dc_belonging), k_doc), dtype=np.float64)
        for d, dc in enumerate(dc_belonging):
            d2dc.update({(d, dc): 1})
        d2dc = d2dc.tocsc()

        # print the internal metrics
        doc_cos_sum = cosine_sum(doc_vectors, k_doc, dc_belonging)
        print 'iter %d - cosine similarity for docs:\tsum %.4f, avg %.4f' % \
              (i + 1, doc_cos_sum, doc_cos_sum / len(dc_belonging))
        word_cos_sum = cosine_sum(doc_vectors.transpose(), k_word, wc_belonging)
        print 'iter %d - cosine similarity for words:\tsum %.4f, avg %.4f' % \
              (i + 1, word_cos_sum, word_cos_sum / len(wc_belonging))

        print '=========================================================='

        w2dc = doc_vectors.transpose() * d2dc

    # return (wc_center, wc_belonging), (dc_center, dc_belonging)
    return (d2wc, dc_belonging), (w2dc, wc_belonging)