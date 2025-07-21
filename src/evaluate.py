# src/evaluate.py
import pandas as pd
import numpy as np
import pickle
import json
import os

def recall_at_k(predicted, actual, k=5):
    """Calculate recall@k metric"""
    predicted_top_k = predicted[:k]
    actual_set = set(actual)
    hits = sum([1 for item in predicted_top_k if item in actual_set])
    return hits / float(min(k, len(actual_set))) if actual_set else 0.0

# File paths
data_file = "data/processed/processed_reviews.csv"
model_file = "models/knn_model.pkl"
output_metrics = "metrics/recall.json"
os.makedirs("metrics", exist_ok=True)

# Load data and model
df = pd.read_csv(data_file)
with open(model_file, 'rb') as f:
    model, user_product_matrix = pickle.load(f)

# Evaluate first 50 users
recall_scores = []
user_ids = user_product_matrix.index.tolist()

for user_id in user_ids[:50]:
    # Get user vector and find similar users
    user_vector = user_product_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=6)
    
    # Get recommendations from similar users (exclude self)
    recommended_items = set()
    for i in indices[0]:
        if user_ids[i] != user_id:
            sim_user_scores = user_product_matrix.iloc[i]
            top_items = sim_user_scores.nlargest(5).index
            recommended_items.update(top_items)
    
    # Get actual items user liked (score >= 4)
    actual_items = df[(df['UserId'] == user_id) & (df['Score'] >= 4)]['ProductId'].tolist()
    
    # Calculate recall
    recall = recall_at_k(list(recommended_items), actual_items, k=5)
    recall_scores.append(recall)

# Save results
avg_recall = np.mean(recall_scores)
metrics = {"recall_at_5": round(avg_recall, 4)}

with open(output_metrics, 'w') as f:
    json.dump(metrics, f)

print(f"Average Recall@5: {avg_recall:.4f}")