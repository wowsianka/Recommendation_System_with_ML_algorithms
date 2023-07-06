
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import dcg_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_prep.dataprep import DataPrep
from itertools import product
import time
from utils.metrics import calculate_dcg, calculate_hit_ratio

data: DataPrep = DataPrep(data_path="data")
reader = Reader(rating_scale=(1,5))
ratings: pd.DataFrame = data.ratings.copy()
ratings.drop('timestamp', axis=1, inplace=True)
ratings.columns = ['user_id', 'item_id', 'rating']
data = Dataset.load_from_df(ratings, reader)
trainset, testset = train_test_split(data,test_size=0.33, random_state=42)


# Define the parameter values to try
# param_values = {
#     'k': [10, 20],
#     'min_k': [3, 5],
#     'name': ['cosine', 'pearson'],
#     'user_based': [True, False]
# }
param_values = {
    'k': [20],
    'min_k': [ 5],
    'name': [ 'pearson'],
    'user_based': [False]
}
param_combinations = list(product(*param_values.values()))

best_rmse = float('inf')
best_params = None
best_fitting_time =None
best_prediction_time = None
best_dcg = 0
# Iterate over parameter combinations
for params in param_combinations:
    algo = KNNWithMeans(k=params[0], min_k=params[1], sim_options={'name': params[2], 'user_based': params[3]})

    start_time = time.time()
    algo.fit(trainset)
    fitting_time = time.time() - start_time

    start_time = time.time()
    predictions = algo.test(testset)
    prediction_time = time.time() - start_time

    rmse = accuracy.rmse(predictions)

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
        
        hit_rate = calculate_hit_ratio(predictions)
        avg_dcg = calculate_dcg(predictions, testset)
        
        if avg_dcg > best_dcg:
            best_dcg = avg_dcg
        best_fitting_time = fitting_time
        best_prediction_time = prediction_time


# Print the best RMSE score and the corresponding best parameters
print(f"Best RMSE score: {best_rmse}")
print("Best parameters:")
for param_name, param_value in zip(param_values.keys(), best_params):
    print(f"{param_name}: {param_value}")

# Print the best Hit Ratio and DCG
print(f"Best Hit Ratio: {hit_rate}")
print(f"Best DCG: {best_dcg}")


# Print the fitting and prediction times
print(f"Fitting time: {fitting_time} seconds")
print(f"Prediction time: {prediction_time} seconds")

