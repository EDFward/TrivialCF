# Collaborative filtering experiments

### Code Structure

- `cf.py`: this is my implementation of various collaborative filtering algorithms.
- `cluster.py`: this script implements reinforcement clustering, which is roughly the same as in the hw2.
- `utility.py`: handles file read/write and data representation.
- `predict.py`: entry point of my program. It runs different CF algorithms with corresponding arguments. `python predict.py` command would rerun all the experiments specified in the handout.
- `README.md`: this document.

### Implementation

`cf.py` is my implementation of collaborative filtering. There are four main function definitions, each corresponds to a specific CF model: `memory_cf`, `model_cf`, `standardization_cf`, and `cluster_cf`. Since their arguments are roughly the same, I think explaining one of them would suffice.

    memory_cf(users, movies, k, similarity_measure, weight_schema,
              repr_matrix, rating_matrix)

This corresponds to the first experiment, using user-user similarity to find nearest neighbors. `users` is a list of user queries, `movies` is a corresponding list of movie queries, `k` denotes the number of nearest neighbors, `similarity_measure` and `weight_schema` are self-explanatory, `repr_matrix` is the data point representation matrix used to search for neighbors and finally the `rating_matrix` is the sparse rating matrix preprocessed as before.

This function first extracts unique users from the user list to avoid duplicate computation when searching for nearest neighbors. Then the distances between those users and all others are calculated and sorted.

Here comes the trick. There are several ways to select the neighbors, and if we simply choose the most similar ones, it’s possible that few of them have actually rated the queried movie thus the scores would tend to be neutral. Therefore my neighbor selection strategy is to select among those who did rate the movie, and it turned out this method produced a significant improvement.

This is also the reason why my running time increases along with value of `k`, since greater `k` would require iterating more of the already-sorted neighbors to find whether they share the same movie rating.

    model_cf(users, movies, k, similarity_measure, weight_schema,
             rating_matrix)

The model-based CF using item-item similarity follows roughly the same procedure except that the movie association matrix is computed offline and serialized into disk. 

I also used the same neighbor selection strategy as in memory-based CF. However as you may notice, model-based CF should theoretically run faster than online memory-based CF because the heavy work (i.e. similarity calculation) has already been done offline, but in my implementation their speed didn’t differ too much. My reason for this phenomenon is that the neighbor selection strategy I use becomes the main bottleneck for the computation, and thus the model-based CF didn’t benefit much from the pre-calculated similarity matrix.

    standardization_cf(users, movies, k, similarity_measure,
                       weight_schema)

This is the PCC-based CF after normalization, which is simply to standardize the rating matrix then use it as the data point representation in memory-based CF. The standardization procedure has already been recorded in previous section. 

    cluster_cf(users, movies, k, similarity_measure, weight_schema,
               cluster_cf_type, k_user, k_movie)

his function used reinforcement clustering to cluster users and movies. In addition to previous arguments it needs `cluster_cf_type` to specify memory or model CF,  along with cluster number of users and movies. The parameter choosing has been discussed in previous parts. 

There are two sub functions under `cluster_cf`, which are `cluster_cf_model` and `cluster_cf_memory`. They are roughly equivalent to the ordinary CF but they don’t need to check whether the neighbor has rated the queried movie because now a neighbor represents a cluster, thus no need to use the previous trick. 

### Design Flaws

The biggest problem of my implementation is code reusability, especially after using the neighbor selection trick. Though the user-user similarity CF and item-item similarity CF should be equivalent with a matrix transposition, I wrote them in separate, which leads to redundant code in different places. Furthermore since the cluster-based CF selected neighbors in different ways (no need to remove itself, didn’t check whether the entry is rated or not, etc.), their code is again separated from the normal memory-/model- based CF.

Despite the code redundancy, there is another problem about flexibility. Using an existing library makes it very easy to develop the algorithm prototype, but on the other hand it increases the difficulty to DIY certain components. For example, during neighbor selection, it’s trivial to only choose neighbors based on similarity because there’s an existing function called `numpy.argpartition`, which is blazingly fast. But to use the proposed neighbor selection strategy I need to tweak a little bit to bypass the slow access of sparse matrix and find efficient ways to iterate NumPy arrays.

There’s always a tradeoff between usability and flexibility.
