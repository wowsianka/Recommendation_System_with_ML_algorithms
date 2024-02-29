import numpy as np
def get_top_k_recs(predictions, k):
    top_k_recs = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_k_recs:
            top_k_recs[uid] = []
        top_k_recs[uid].append((iid, est))
    for uid, user_ratings in top_k_recs.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_k_recs[uid] = user_ratings[:k]
    return top_k_recs

def calculate_hit_ratio(predictions, testset, k):
    top_k_recs = get_top_k_recs(predictions, k)
    total = 0
    hit_rate = 0
    for uid, recs in top_k_recs.items():
        true_items = set([iid for (uid_test, iid, _) in testset if uid_test == uid])
        if any(item in true_items for item in recs):
            hit_rate += 1
        hit_rate /= len(top_k_recs)
    hit_rate /= len(top_k_recs)
    return hit_rate

def calculate_dcg(predictions, testset, k):
    top_k_recs = get_top_k_recs(predictions, k)
    dcg = 0
    for uid, recs in top_k_recs.items():
        test_items = [iid for (uid, iid, r) in testset if uid == uid]
        relevance = [1 if rec[0] in test_items else 0 for rec in recs]
        user_dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
        dcg += user_dcg
    avg_dcg = dcg / len(top_k_recs)
    return avg_dcg


import math
def calc_avg_dcg(predictions, test, num_users=100, top_n=15, min_pred_rating=4):
    # Convert test to a dictionary for faster lookup
    test_dict = { (user, movie): rating for user, movie, rating in test }

    # Group predictions by user
    predictions_by_user = {}
    for userId, movieId, real_rating, predicted_rating, _ in predictions:
        if userId not in predictions_by_user:
            predictions_by_user[userId] = []
        predictions_by_user[userId].append((userId, movieId, real_rating, predicted_rating))

    # Choose num_users users at random
    user_ids = list(predictions_by_user.keys())
    chosen_users = np.random.choice(user_ids, num_users, replace=False)

    # Initialize DCG sum
    dcg_sum = 0

    # Calculate DCG for each chosen user
    for user in chosen_users:
        predictions = predictions_by_user[user]

        # Only consider predictions with a predicted rating of min_pred_rating or higher
        predictions = [p for p in predictions if p[3] >= min_pred_rating]

        # Sort predictions by predicted rating in descending order
        predictions.sort(key=lambda x: -x[3])

        # Limit to top top_n predictions
        predictions = predictions[:top_n]

        # Create a relevance list
        rel = [1 if (user, movieId) in test_dict else 0 for userId, movieId, _, _ in predictions]

        # Calculate DCG for this user
        dcg = sum(1/np.log2(i+2) for i, value in enumerate(rel) if value == 1)

        # Add this user's DCG to the sum
        dcg_sum += dcg

    # Calculate average DCG
    avg_dcg = dcg_sum / num_users

    return avg_dcg