import time
import re
import requests
import sys
from cf import memory_cf, model_cf, standardization_cf, cluster_cf


def predict(output_file_path, test_file_path='../dev.csv', cf_type='memory',
            k=10, similarity_measure='cosine', weight_schema='mean',
            cluster_cf_type=None):
    # argument sanity check
    if (similarity_measure not in ('cosine', 'dot_product') or
            weight_schema not in ('mean', 'weighted_mean')):
        print '==ERROR== unsupported arguments for model CF'
        sys.exit(1)

    test_movies, test_users = [], []
    with open(test_file_path) as f:
        for line in f:
            movie, user = map(int, line.split(','))
            test_movies.append(movie)
            test_users.append(user)
    if cf_type == 'memory':
        ratings = memory_cf(test_users, test_movies, k, similarity_measure,
                            weight_schema)
    elif cf_type == 'model':
        ratings = model_cf(test_users, test_movies, k, similarity_measure,
                           weight_schema)
    elif cf_type == 'pcc':
        ratings = standardization_cf(test_users, test_movies, k,
                                     similarity_measure, weight_schema)
    elif cf_type == 'cluster':
        ratings = cluster_cf(test_users, test_movies, k, similarity_measure,
                             weight_schema, cluster_cf_type, 300, 150)
    else:
        print '==ERROR== wrong arguments for predictions'
        return

    with open(output_file_path, 'wb') as f:
        for rating in ratings:
            f.write(str(rating) + '\n')


if __name__ == '__main__':
    # experiment time!
    URL = 'http://nyc.lti.cs.cmu.edu/classes/11-741/s15/HW/HW4/eval/upload.cgi'
    OUTPUT_FILE = '../dev-result.txt'
    SCORE_PATTERN = re.compile('/H4>(.*)<BR')

    ####
    #### Experiment 2.1
    ####

    for sim in ('dot_product', 'cosine'):
        for k in (10, 100, 500):
            start = time.time()
            predict(OUTPUT_FILE, cf_type='memory', k=k,
                    similarity_measure=sim, weight_schema='mean')
            runtime = time.time() - start
            response = requests.post(URL,
                                     files={'infile': open(OUTPUT_FILE, 'rb')})
            rmse = SCORE_PATTERN.search(response.text).group(1)
            print 'mean, %s, k=%d, rmse:%s, time:%f' % (sim, k, rmse, runtime)

    for k in (10, 100, 500):
        start = time.time()
        predict(OUTPUT_FILE, cf_type='memory', k=k,
                similarity_measure='cosine', weight_schema='weighted_mean')
        runtime = time.time() - start
        response = requests.post(URL,
                                 files={'infile': open(OUTPUT_FILE, 'rb')})
        rmse = SCORE_PATTERN.search(response.text).group(1)
        print 'weighted_sum, cosine, k=%d, rmse:%s, time:%f' % (k,
                                                                rmse, runtime)

    ####
    #### Experiment 2.2
    ####

    for sim in ('dot_product', 'cosine'):
        for k in (10, 100, 500):
            start = time.time()
            predict(OUTPUT_FILE, cf_type='model', k=k,
                    similarity_measure=sim, weight_schema='mean')
            runtime = time.time() - start
            response = requests.post(URL,
                                     files={'infile': open(OUTPUT_FILE, 'rb')})
            rmse = SCORE_PATTERN.search(response.text).group(1)
            print 'mean, %s, k=%d, rmse:%s, time:%f' % (sim, k, rmse, runtime)
    
    for k in (10, 100, 500):
        start = time.time()
        predict(OUTPUT_FILE, cf_type='model', k=k,
                similarity_measure='cosine', weight_schema='weighted_mean')
        runtime = time.time() - start
        response = requests.post(URL,
                                 files={'infile': open(OUTPUT_FILE, 'rb')})
        rmse = SCORE_PATTERN.search(response.text).group(1)
        print 'weighted_sum, cosine, k=%d, rmse:%s, time:%f' % (k,
                                                                rmse, runtime)

    ####
    #### Experiment 2.3
    ####

    for weight in ('mean', 'weighted_mean'):
        for k in (10, 100, 500):
            start = time.time()
            predict(OUTPUT_FILE, cf_type='pcc', k=k,
                    similarity_measure='cosine', weight_schema=weight)
            runtime = time.time() - start
            response = requests.post(URL,
                                     files={'infile': open(OUTPUT_FILE, 'rb')})
            rmse = SCORE_PATTERN.search(response.text).group(1)
            print '%s, cosine, k=%d, rmse:%s, time:%f' % (weight, k, rmse,
                                                          runtime)

    ####
    #### Experiment 2.5
    ####

    for sim in ('dot_product', 'cosine'):
        for k in (10, 100,):
            predict(OUTPUT_FILE, cf_type='cluster', k=k,
                    similarity_measure=sim, weight_schema='mean',
                    cluster_cf_type='memory')
            response = requests.post(URL,
                                     files={'infile': open(OUTPUT_FILE, 'rb')})
            rmse = SCORE_PATTERN.search(response.text).group(1)
            print 'mean, %s, k=%d, rmse:%s' % (sim, k, rmse)

    for k in (10, 100,):
        predict(OUTPUT_FILE, cf_type='cluster', k=k,
                similarity_measure='cosine', weight_schema='weighted_mean',
                cluster_cf_type='memory')
        response = requests.post(URL,
                                 files={'infile': open(OUTPUT_FILE, 'rb')})
        rmse = SCORE_PATTERN.search(response.text).group(1)
        print 'weighted_sum, cosine, k=%d, rmse:%s' % (k, rmse)

    ####
    #### Experiment 2.6
    ####

    for sim in ('dot_product', 'cosine'):
        for k in (10, 100,):
            predict(OUTPUT_FILE, cf_type='cluster', k=k,
                    similarity_measure=sim, weight_schema='mean',
                    cluster_cf_type='model')
            response = requests.post(URL,
                                     files={'infile': open(OUTPUT_FILE, 'rb')})
            rmse = SCORE_PATTERN.search(response.text).group(1)
            print 'mean, %s, k=%d, rmse:%s' % (sim, k, rmse)

    for k in (10, 100,):
        predict(OUTPUT_FILE, cf_type='cluster', k=k,
                similarity_measure='cosine', weight_schema='weighted_mean',
                cluster_cf_type='model')
        response = requests.post(URL,
                                 files={'infile': open(OUTPUT_FILE, 'rb')})
        rmse = SCORE_PATTERN.search(response.text).group(1)
        print 'weighted_sum, cosine, k=%d, rmse:%s' % (k, rmse)
