from cf import memory_cf, model_cf, standardization_cf, cluster_cf


def predict(output_file_path, test_file_path='../dev.csv', cf_type='memory',
            k=10, similarity_measure='cosine', weight_schema='mean',
            cluster_cf_type=None):
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
                             weight_schema, cluster_cf_type, 200, 100)
    else:
        print '==ERROR== wrong arguments for predictions'
        return

    with open(output_file_path, 'wb') as f:
        for rating in ratings:
            f.write(str(rating) + '\n')


if __name__ == '__main__':
    predict('../dev-result.txt', cf_type='cluster', k=20,
            similarity_measure='cosine', weight_schema='weighted_mean',
            cluster_cf_type='memory')
