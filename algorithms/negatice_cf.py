import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_prep.dataprep import DataPrep
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Input, Dropout, Dense, BatchNormalization, Concatenate, Dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from datetime import datetime
import pandas as pd
from typing import List
import re
import numpy as np
from surprise import Reader, Dataset, NMF, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split, cross_validate
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF as sklearn_nmf

from annoy import AnnoyIndex

def movie_use_matrix_pivot(df_):
    mu_matrix = df_.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    mu_matrix_cp = csr_matrix(mu_matrix.values)
    return mu_matrix, mu_matrix_cp

def get_item_latent_factor(dimension, matrix_cp):
    nmf_model = sklearn_nmf(n_components=dimension)
    item_vectors = nmf_model.fit_transform(matrix_cp)
    return item_vectors


def get_nmf_rmse(sample_data, num_factors):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(sample_data[['userId', 'movieId', 'rating']], reader)
    trainset, testset = surprise_train_test_split(data, test_size=0.2)
    algo = NMF(n_factors=num_factors)
    
    start_time = time.time()
    fitted_algo = algo.fit(trainset)
    fitting_time = time.time() - start_time

    start_time = time.time()
    predictions = fitted_algo.test(testset)
    prediction_time = time.time() - start_time

    rmse_value = accuracy.rmse(predictions)
    return rmse_value, fitting_time, prediction_time


def ann(matrix, item_vectors, metric, num_trees):
    '''
    Implement Approximate Nearest Neighborhood to find similar items, save it in 'rating.ann'
    input: target movie, rating matrix, item_vectors, metric (can be "angular", "euclidean", "manhattan", "hamming")
    number of trees(More trees gives higher precision when querying)
    output: save it in 'rating.ann'
    '''

    # construct a dictionary where movied id contains its vector representation
    movieids = matrix.columns.tolist()
    rating_dictionary = {movieids[i]: item_vectors[i] for i in range(len(movieids))}
    # ann method
    f = len(item_vectors[1])
    t = AnnoyIndex(f, metric)  # Length of item vector that will be indexed
    for key in rating_dictionary:
        t.add_item(key, rating_dictionary.get(key))
    t.build(num_trees) # 10 trees
    t.save('rating.ann')


def get_similar_movies(f, target, metric, top_n):
    '''
    Find similar items given the targeted movies
    input: f is the length of item vector that will be indexed, top_n - n closed movies
    output: a 2 element tuple with two lists in it,
    the first one top_n similar movies from closer to further away from the target movies
    the second one containing all corresponding distances
    '''
    u = AnnoyIndex(f, metric)
    u.load('rating.ann') # super fast, will just mmap the file
    movies, distances = u.get_nns_by_item(target, top_n, include_distances=True)
    return movies[1:], distances[1:]


def get_rated_movies(data, userid, threshold=2):
    '''
    input: rating dataset, userid, a rating threshold, movies that are rated below threshold
    will not be counted
    output: a list of high-scored movies that are rated by givern user, a list of corresponding ratings
    '''
    all_rates = data[data['userId'] == userid]
    high_rates = all_rates[all_rates['rating'] >= threshold]['rating'].values
    high_rate_movie = all_rates[all_rates['rating'] >= threshold]['movieId'].values
    return high_rate_movie, high_rates

def get_recommendation(data, dimension, matrix_cp, matrix, metric, num_tree, threshold, top_n):
    '''
    Get recommendation list for each user in the data set
    input: data - orginal dataframe; dimension - number of latent factors in NMF; matrix_cp - compressed rating matrix;
           matrix - rating matrix; metric - distance metric in ANN method; num_tree - number of trees in ANN;
           threshold - rating threshold; top_n - most n similar movies
    output: return a recommendation lsit for each user, each list consists of at most 20 movies.
    '''
    userIds = data.userId.unique()
    v = get_item_latent_factor(dimension, rating_matrix_cp)
    ann(matrix, v, metric, num_tree)     # save the ann in 'rating.ann' file
    f = len(v[1])
    u = AnnoyIndex(f, metric)
    u.load('rating.ann')
    # construct the recommendation list for each user
    recommendation_dict = {}
    for userid in userIds:
        high_rate_movie, rate = get_rated_movies(data, userid, threshold)
        movielist = []
        distancelist = []
        if len(high_rate_movie) > 1:
            # find neighborhood of each movies in the high rated movie set
            for movieid in high_rate_movie:
                movie, dist = u.get_nns_by_item(movieid, top_n, include_distances=True)
                movielist.extend(movie[1:])
                # get the weighted distance based on rating scores
                weighted_dist = (np.array(dist[1:])/rate[np.where(high_rate_movie == movieid)]).tolist()
                distancelist.extend(weighted_dist)
            #if more than 20 movies are chosen to recommend to user, choose 20 nearest item for this user
            if len(movielist) > 20:
                sorted_recommend = np.array(movielist)[np.array(distancelist).argsort()]
                movielist = sorted_recommend[:20]
        recommendation_dict[userid] = movielist
    return recommendation_dict


    def get_top_k_recommendations(u, high_rate_movies, ratings, top_k):
    movielist = []
    distancelist = []
    for movieid, rating in zip(high_rate_movies, ratings):
        movie, dist = u.get_nns_by_item(movieid, top_k, include_distances=True)
        movielist.extend(movie)
        weighted_dist = (np.array(dist) / rating).tolist()
        distancelist.extend(weighted_dist)
    sorted_recommend = np.array(movielist)[np.array(distancelist).argsort()]
    top_k_recommendations = sorted_recommend[:top_k].tolist()
    return top_k_recommendations

def get_hitrate(data, dimension, matrix_cp, matrix, metric, num_tree, threshold, top_k):
    userIds = data.userId.unique()
    v = get_item_latent_factor(dimension, rating_matrix_cp)
    ann(matrix, v, metric, num_tree)     # save the ann in 'rating.ann' file
    f = len(v[1])
    u = AnnoyIndex(f, metric)
    u.load('rating.ann')
    hit_count = 0
    total_count = 0
    for userid in userIds:
        high_rate_movies, ratings = get_rated_movies(data, userid, threshold)
        if len(high_rate_movies) >= top_k:
            recs = get_top_k_recommendations(u, high_rate_movies, ratings, top_k)
            true_items = set(high_rate_movies)
            intersection_count = len(set(recs).intersection(true_items))
            if intersection_count >= top_k:
                hit_count += 1
            total_count += 1
    hit_rate = hit_count / total_count if total_count > 0 else 0
    return hit_rate

    
def calculate_dcg(data, dimension, matrix_cp, matrix, metric, num_tree, threshold, top_k):
    userIds = data.userId.unique()
    v = get_item_latent_factor(dimension, rating_matrix_cp)
    ann(matrix, v, metric, num_tree)     # save the ann in 'rating.ann' file
    f = len(v[1])
    u = AnnoyIndex(f, metric)
    u.load('rating.ann')

    total_dcg = 0
    total_users = 0
    for userid in userIds:
        high_rate_movies, ratings = get_rated_movies(data, userid, threshold)
        if len(high_rate_movies) >= top_k:
            recs = get_top_k_recommendations(u, high_rate_movies, ratings, top_k)
            true_items = set(high_rate_movies)

            # Calculate relevance (binary relevance in this case)
            relevance = [1 if rec in true_items else 0 for rec in recs]

            # Calculate DCG for the user
            user_dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])

            total_dcg += user_dcg
            total_users += 1

    # Calculate average DCG
    avg_dcg = total_dcg / total_users if total_users > 0 else 0
    return avg_dcg

# one set of hyperparameter output below dictionary
# res = get_recommendation(sample_dft, 20, rating_matrix_cp, rating_matrix, 'angular', 10, 2, 10)




# Load the data and split it
data: DataPrep = DataPrep(data_path="data")
reader = Reader(rating_scale=(1,5))
ratings: pd.DataFrame = data.ratings.copy()
ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']

sample_df = ratings
# Userid, movieid encoding by indices
user_ids = sample_df['user_id'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
sample_df['user'] = sample_df['user_id'].map(user2user_encoded)

item_ids = sample_df['item_id'].unique().tolist()
item2item_encoded = {x: i for i, x in enumerate(item_ids)}
sample_df['item'] = sample_df['item_id'].map(item2item_encoded)

# Train-test split
train, test = train_test_split(sample_df, test_size = 0.2, random_state=123)

# Get RMSE on the test set
# nmf_rmse, fitting_time, prediction_time= get_nmf_rmse(sample_dft, 15)
# print("RMSE:", nmf_rmse)

# # Print the fitting and prediction times
# print(f"Fitting time: {fitting_time} seconds")
# print(f"Prediction time: {prediction_time} seconds")

############### Get mean_scores for different n factors
# mean_scores = []
# reader = Reader(rating_scale=(0, 5))
# data = Dataset.load_from_df(sample_dft[['userId', 'movieId', 'rating']], reader)

# for n in [ 14, 15, 16, 17, 18]:
#     nmf_model = NMF(n_factors = n)
#     score = cross_validate(nmf_model, data, measures=['RMSE'], cv=3, verbose=False)['test_rmse']
#     mean_scores.append(score.mean())
# mean_scores
###############



trees = [10, 15, 20, 25, 30]
dcg_list = []
for tree in trees:
    h = calculate_dcg(test, 18, rating_matrix_cp, rating_matrix, 'angular', tree, 3, 10)
    dcg_list.append(h)
dcg_list