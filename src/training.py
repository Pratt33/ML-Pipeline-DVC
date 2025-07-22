# src/train.py
import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def load_parameters(path="params.yaml"):
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return params

def load_file_paths(data_file, output_dir, output_model):
    os.makedirs(output_dir, exist_ok=True)
    output_model = os.path.join(output_dir, output_model)
    return data_file, output_model


def use_parameters():
    params = load_parameters()
    min_user_interactions = params["train"]["min_user_interactions"]
    min_product_interactions = params["train"]["min_product_interactions"]
    metric = params["train"]["metric"]
    n_neighbors = params["train"]["neighbors"]
    return min_user_interactions, min_product_interactions, metric, n_neighbors

def filter_data(df, min_user_interactions, min_product_interactions):
    user_counts = df['UserId'].value_counts()
    product_counts = df['ProductId'].value_counts()
    active_users = user_counts[user_counts >= min_user_interactions].index
    active_products = product_counts[product_counts >= min_product_interactions].index

    return df[df['UserId'].isin(active_users) & df['ProductId'].isin(active_products)]
# Filter active users/products

def extract_top_summaries(df):
    # Extract top summary for each product with Score >= 4
    positive_reviews = df[df['Score'] >= 4]
    top_summaries = (
        positive_reviews
        .groupby("ProductId")["Summary"]
        .first()
        .fillna("No summary available")
        .to_dict()
    )
    return top_summaries

def create_matrix(df):
    user_product_matrix = df.pivot_table(index='UserId', columns='ProductId', values='Score').fillna(0)
    return user_product_matrix
# Create user-item matrix

def train_model(user_product_matrix, metric, n_neighbors):
    sparse_matrix = csr_matrix(user_product_matrix.values)
    model = NearestNeighbors(metric=metric, algorithm='brute', n_neighbors=n_neighbors)
    model.fit(sparse_matrix)
    return model
# Train KNN model
# Save model, matrix, and summaries
def save_model(model, user_product_matrix, top_summaries, output_model):
    with open(output_model, 'wb') as f:
        pickle.dump((model, user_product_matrix, top_summaries), f)
    return output_model

def main():
    # Define file paths
    data_file = "data/processed/processed_reviews.csv"
    output_dir = "models"
    
    df = pd.read_csv(data_file)

    os.makedirs(output_dir, exist_ok=True)
    
    data_file, output_model = load_file_paths(data_file, output_dir, "knn_model.pkl")
    min_user_interactions, min_product_interactions, metric, n_neighbors = use_parameters()

    df = filter_data(df, min_user_interactions, min_product_interactions)
    top_summaries = extract_top_summaries(df)
    user_product_matrix = create_matrix(df)

    model = train_model(user_product_matrix, metric, n_neighbors)
    output_model = save_model(model, user_product_matrix, top_summaries, output_model)
    #print(f"Model saved to: {output_model}")
    

if __name__ == "__main__":
    main()