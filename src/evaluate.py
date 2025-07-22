# src/evaluate.py
import pandas as pd
import numpy as np
import pickle
import json
import os
import yaml
import sys

def recall_at_k(predicted: list, actual: list, k: int = 5) -> float:
    try:
        predicted_top_k = predicted[:k]
        actual_set = set(actual)
        hits = sum([1 for item in predicted_top_k if item in actual_set])
        return hits / float(min(k, len(actual_set))) if actual_set else 0.0
    except Exception as e:
        print(f"Error calculating recall@k: {e}")
        return 0.0

def load_param_file() -> dict:
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        print("Error: params.yaml file not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

def load_parameters() -> tuple:
    try:
        params = load_param_file()
        top_k = params["evaluate"]["top_k"]
        eval_user_count = params["evaluate"]["eval_user_count"]
        rating_threshold = params["evaluate"]["rating_threshold"]
        return top_k, eval_user_count, rating_threshold
    except KeyError as e:
        print(f"Error: Missing evaluation parameter {e} in params.yaml")
        sys.exit(1)

def load_file_paths() -> tuple:
    try:
        data_file = "data/processed/processed_reviews.csv"
        model_file = "models/knn_model.pkl"
        output_metrics = "metrics/recall.json"
        os.makedirs("metrics", exist_ok=True)
        return data_file, model_file, output_metrics
    except Exception as e:
        print(f"Error setting up file paths: {e}")
        sys.exit(1)

def load_model_data(model_file):
    try:
        with open(model_file, 'rb') as f:
            model, user_product_matrix, top_summaries = pickle.load(f)
        return model, user_product_matrix, top_summaries
    except FileNotFoundError:
        print(f"Error: Model file {model_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def evaluate_model(model, user_product_matrix: pd.DataFrame, df: pd.DataFrame, top_k: int, eval_user_count: int, rating_threshold: int) -> dict:
    try:
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

        avg_recall = np.mean(recall_scores) if recall_scores else 0.0
        metrics = {"recall_at_5": round(avg_recall, 4)}

        return metrics
    except Exception as e:
        print(f"Error evaluating model: {e}")
        sys.exit(1)

def output_metrics(metrics: dict, output_metrics: str) -> str:
    try:
        with open(output_metrics, 'w') as f:
            json.dump(metrics, f)
        return output_metrics
    except Exception as e:
        print(f"Error saving metrics to {output_metrics}: {e}")
        sys.exit(1)

def main():
    try:
        top_k, eval_user_count, rating_threshold = load_parameters()
        data_file, model_file, output_metrics_path = load_file_paths()
        model, user_product_matrix, top_summaries = load_model_data(model_file)
        df = pd.read_csv(data_file)

        metrics = evaluate_model(model, user_product_matrix, df, top_k, eval_user_count, rating_threshold)
        output_metrics(metrics, output_metrics_path)
        print(f"Model evaluation completed. Metrics saved to: {output_metrics_path}")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()