import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from surprise import Reader
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
from utils.metrics import calculate_dcg, calculate_hit_ratio
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel

class NCFHyperModel(HyperModel):
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

    def build(self, hp):
        # Input layers
        user = Input(shape=(1,), name='user')
        item = Input(shape=(1,), name='item')

        # Embedding size
        embed_size = hp.Int('embed_size', min_value=10, max_value=50, step=5)

        # MLP Embeddings
        user_mlp_embedding = Embedding(input_dim=self.num_users, output_dim=embed_size, name='user_mlp_embedding')(user)
        item_mlp_embedding = Embedding(input_dim=self.num_items, output_dim=embed_size, name='item_mlp_embedding')(item)

        # MF Embeddings
        user_mf_embedding = Embedding(input_dim=self.num_users, output_dim=embed_size, name='user_mf_embedding')(user)
        item_mf_embedding = Embedding(input_dim=self.num_items, output_dim=embed_size, name='item_mf_embedding')(item)

        # MLP layers
        user_mlp_embedding_flat = Flatten()(user_mlp_embedding)
        item_mlp_embedding_flat = Flatten()(item_mlp_embedding)
        mlp_concat = Concatenate()([user_mlp_embedding_flat, item_mlp_embedding_flat])

        mlp_layer = Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu')(mlp_concat)
        mlp_dropout = Dropout(rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1))(mlp_layer)

        # MF layer
        user_mf_embedding_flat = Flatten()(user_mf_embedding)
        item_mf_embedding_flat = Flatten()(item_mf_embedding)
        mf_multiply = Dot(axes=1)([user_mf_embedding_flat, item_mf_embedding_flat])

        # Combine MLP and MF
        combine_mlp_mf = Concatenate()([mlp_dropout, mf_multiply])

        # Result layer
        result = Dense(units=1)(combine_mlp_mf)

        model = Model(inputs=[user, item], outputs=result)
        model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
        return model

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



hypermodel = NCFHyperModel(num_users=num_users, num_items=num_items)

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='ncf')

tuner.search_space_summary()

# Start timing
start_time = time.time()
# Then, use your data to search the best hyperparameters.
# For example:
tuner.search([train['user'].values, train['item'].values],
              train['rating'].values,
              validation_data=([test['user'].values, test['item'].values], test['rating'].values),
              epochs=5)

tuner.results_summary()

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

start_time = time.time()
# Fit the best model
best_model.fit([train['user'].values, train['item'].values], train['rating'].values, epochs=5)
fitting_time = time.time() - start_time

# Print the fitting and prediction times
print(f"Fitting time: {fitting_time} seconds")

best_model.summary()