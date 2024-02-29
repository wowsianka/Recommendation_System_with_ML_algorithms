from surprise import BaselineOnly
from surprise import AlgoBase, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_prep.dataprep import DataPrep
from utils.metrics import calculate_dcg, calculate_hit_ratio, calc_avg_dcg
from data_prep.dataprep import DataPrep
import time
import pandas as pd

'''
 BaselineOnly algorithm from the surprise library. 
 This algorithm predicts the baseline estimate for given user and item. 
 The baseline estimate of an item for a user is the sum of the overall mean rating, 
 the user bias (average deviation from the mean for the specific user) 
 and the item bias (average deviation from the mean for the specific item).
 Here, we use Stochastic Gradient Descent ('sgd') for optimization, which is a common method for training machine learning models. The learning rate is set to a small number, which is the step size at each iteration while moving toward a minimum of a loss function. It's a hyperparameter that controls how much we are adjusting the weights of our network concerning the loss gradient.

 Remember that the optimal learning rate might vary depending on the data, so you may want to tune this parameter to get the best results. The biases (user and item) are initialized to 0, and the optimization procedure will find the best values.

 In terms of comparing recommendation algorithms, BaselineOnly serves as a good 
 baseline as it simply uses the average ratings and biases, not considering other 
 factors like user-item interactions or item content. Other recommendation algorithms 
 should be able to achieve better performance by making use of additional information.
'''


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
trainset, testset = train_test_split(data,test_size=0.33, random_state=42)

# Create BaselineOnly recommender instance
bsl_options = {'method': 'sgd',
               'learning_rate': .00005,
               }

baseline_recommender = BaselineOnly(bsl_options=bsl_options)

start_time = time.time()
baseline_recommender.fit(trainset)
fitting_time = time.time() - start_time

start_time = time.time()
predictions = baseline_recommender.test(testset)
prediction_time = time.time() - start_time


# Print the fitting and prediction times
print(f"Fitting time: {fitting_time} seconds")
print(f"Prediction time: {prediction_time} seconds")


# Then compute RMSE
rmse = accuracy.rmse(predictions)
# hit_rate = calculate_hit_ratio(predictions, testset, 5)
# avg_dcg = calculate_dcg(predictions, testset, 5)
ndcg = calc_avg_dcg(predictions, testset)

print(f"RMSE score: {rmse}")
print(f"NDCg:{ndcg}")


###########################-----RESULTS--------###################
# Fitting time: 4.3900134563446045 seconds
# Prediction time: 2.701997995376587 seconds
# RMSE: 1.0216
# RMSE score: 1.0216422437190105
#  Hit Ratio: 0.991225165562914
#  DCG: 2.9424222564952736