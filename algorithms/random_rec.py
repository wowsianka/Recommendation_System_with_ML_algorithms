from surprise import AlgoBase, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_prep.dataprep import DataPrep
from utils.metrics import calc_avg_dcg
import time
import pandas as pd

class RandomRecommender(AlgoBase):
    '''
    This class inherits from AlgoBase and implements a fit() method 
    (which doesn't do much, as our RandomRecommender doesn't require any training), 
    and an estimate() method, which provides the prediction. 
    For the RandomRecommender, the estimate method just returns the global mean rating.
    This will provide a random prediction based on the global mean of the ratings, serving as a baseline to compare with more complex recommendation systems. 
    '''
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        return self

    def estimate(self, u, i):
        return self.trainset.global_mean
    


data: DataPrep = DataPrep(data_path="data")

# Load the data
reader = Reader(rating_scale=(1,5))
ratings: pd.DataFrame = data.ratings.copy()

# Drop unnecessary columns
ratings.drop('timestamp', axis=1, inplace=True)
ratings.columns = ['user_id', 'item_id', 'rating']

# Load from df
data = Dataset.load_from_df(ratings, reader)

# Split the data
trainset, testset = train_test_split(data,test_size=0.2, random_state=42)

# Create a RandomRecommender instance
random_recommender = RandomRecommender()

# Train the algorithm on the trainset, and predict ratings for the testset

start_time = time.time()
random_recommender.fit(trainset)
fitting_time = time.time() - start_time

start_time = time.time()
predictions = random_recommender.test(testset)
prediction_time = time.time() - start_time


# Print the fitting and prediction times
print(f"Fitting time: {fitting_time} seconds")
print(f"Prediction time: {prediction_time} seconds")


# Then compute RMSE
rmse = accuracy.rmse(predictions)
ndcg = calc_avg_dcg(predictions, testset)

print(f"RMSE score: {rmse}")
print(f"NDCg:{ndcg}")
