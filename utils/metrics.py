import numpy as np

def get_top_k_recs(predictions):
    top_k = 5
    top_k_recs = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_k_recs:
            top_k_recs[uid] = []
        top_k_recs[uid].append((iid, est))
    return top_k, top_k_recs

def calculate_hit_ratio(predictions):
    top_k, top_k_recs = get_top_k_recs(predictions)
    hit_rate = 0
    for uid, recs in top_k_recs.items():
        true_items = set([iid for iid, _ in recs])
        if len(true_items) >= top_k:
            hit_rate += 1

    hit_rate /= len(top_k_recs)
    return hit_rate


def calculate_dcg(predictions, testset):
    top_k, top_k_recs = get_top_k_recs(predictions)
    dcg = 0
    for uid, recs in top_k_recs.items():
        # Sort items based on estimated rating
        recs.sort(key=lambda x: x[1], reverse=True)
        recs = recs[:top_k]
        
        # Check if items are in the test set for the user
        test_items = [iid for (uid, iid, r) in testset if uid == uid]
        
        # Calculate relevance
        relevance = [1 if rec[0] in test_items else 0 for rec in recs]
        
        # Calculate DCG for the user
        user_dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
        dcg += user_dcg

    # Calculate average DCG
    avg_dcg = dcg / len(top_k_recs)
    return avg_dcg