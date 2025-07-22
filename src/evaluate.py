# src/evaluate.py
import pandas as pd
import numpy as np
import pickle
import json
import os
import yaml

def recall_at_k(predicted, actual, k=5):
    predicted_top_k = predicted[:k]
    actual_set = set(actual)
    hits = sum([1 for item in predicted_top_k if item in actual_set])
    return hits / float(min(k, len(actual_set))) if actual_set else 0.0

def load_param_file():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params
# Load parameters
def load_parameters():
    params = load_param_file()
    # Extract parameters for evaluation
    top_k = params["evaluate"]["top_k"]
    eval_user_count = params["evaluate"]["eval_user_count"]
    rating_threshold = params["evaluate"]["rating_threshold"]
    return top_k, eval_user_count, rating_threshold


# File paths
def load_file_paths():
    data_file = "data/processed/processed_reviews.csv"
    model_file = "models/knn_model.pkl"
    output_metrics = "metrics/recall.json"
    os.makedirs("metrics", exist_ok=True)
    return data_file, model_file, output_metrics

def load_model_data(model_file):
    with open(model_file, 'rb') as f:
        model, user_product_matrix, top_summaries = pickle.load(f)
    return model, user_product_matrix, top_summaries
# Load data and model
df = pd.read_csv(load_file_paths()[0])

# Evaluate
def evalute_model(model, user_product_matrix, df, top_k, eval_user_count, rating_threshold):

    recall_scores = []
    user_ids = user_product_matrix.index.tolist()

    for user_id in user_ids[:eval_user_count]:
        user_vector = user_product_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = model.kneighbors(user_vector, n_neighbors=top_k + 1)

        recommended_items = set()
        for i in indices[0]:
            if user_ids[i] != user_id:
                sim_user_scores = user_product_matrix.iloc[i]
                top_items = sim_user_scores.nlargest(top_k).index
                recommended_items.update(top_items)

        actual_items = df[(df['UserId'] == user_id) & (df['Score'] >= rating_threshold)]['ProductId'].tolist()
        recall = recall_at_k(list(recommended_items), actual_items, k=top_k)
        recall_scores.append(recall)

    avg_recall = np.mean(recall_scores)
    metrics = {"recall_at_5": round(avg_recall, 4)}

    return metrics

def output_metrics(metrics, output_metrics):
    with open(output_metrics, 'w') as f:
        json.dump(metrics, f)
    return output_metrics

def main():
    top_k, eval_user_count, rating_threshold = load_parameters()
    data_file, model_file, output_metrics_path = load_file_paths()
    model, user_product_matrix, top_summaries = load_model_data(model_file)
    df = pd.read_csv(data_file)

    metrics = evalute_model(model, user_product_matrix, df, top_k, eval_user_count, rating_threshold)
    output_metrics(metrics, output_metrics_path)

if __name__ == "__main__":
    main()