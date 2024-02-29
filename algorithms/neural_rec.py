import numpy as np
from surprise import Reader
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
from utils.metrics import calc_avg_dcg
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model

# Load the data and split it
data: DataPrep = DataPrep(data_path="data")
reader = Reader(rating_scale=(1,5))
ratings: pd.DataFrame = data.ratings.copy()
ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']

# All unique user ids
user_ids = ratings["user_id"].unique().tolist() 
num_all_user = len(user_ids)

# Randomly select 20% users from rating dataset 
np.random.seed(123)
rand_userid = np.random.choice(user_ids, size = int(num_all_user * 0.1), replace=False)
sample_df = ratings.loc[ratings['user_id'].isin(rand_userid)]

# Userid, movieid encoding by indices
user_ids = sample_df['user_id'].unique().tolist()
num_users = len(user_ids)
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
sample_df['user'] = sample_df['user_id'].map(user2user_encoded)

item_ids = sample_df['item_id'].unique().tolist()
num_items = len(item_ids)
item2item_encoded = {x: i for i, x in enumerate(item_ids)}
sample_df['item'] = sample_df['item_id'].map(item2item_encoded)

# Train-test split
train, test = train_test_split(sample_df, test_size = 0.2, random_state=123)


best_model = load_model('models/best_neural_model.h5')

# Generate predictions
test_user_item_array = [test['user'].values, test['item'].values]


predictions = best_model.predict(test_user_item_array)

# Flatten the predictions array
start_time = time.time()
predictions = predictions.flatten()
prediction_time = time.time() - start_time



print(f"Prediction time: {prediction_time} seconds")


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test['rating'].values, predictions))
print(f'Test RMSE: {rmse}')

# Add predictions to the test dataframe
test['predictions'] = predictions

# Create list of tuples in the format (user_id, item_id, rating, prediction, None)
predictions_list = [(user_id, item_id, rating, pred, None) for user_id, item_id, rating, pred in test[['user_id', 'item_id', 'rating', 'predictions']].itertuples(index=False, name=None)]

testset_list = list(test[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))

# Calculate DCG
dcg = calc_avg_dcg(predictions_list, testset_list)
print(f'DCG: {dcg}')


####---RESULTS:
# Test RMSE: 0.7600139030345904
# Hit Ratio: 0.9271523178807947
# DCG: 2.8944902987098016