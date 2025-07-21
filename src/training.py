# src/train.py
import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load parameters - FIX: Pass the file object to yaml.safe_load()
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

input_file = "data/processed/processed_reviews.csv"
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
output_model = os.path.join(output_dir, "knn_model.pkl")

df = pd.read_csv(input_file)

# Use parameters
min_user_interactions = params["train"]["min_user_interactions"]
min_product_interactions = params["train"]["min_product_interactions"]
metric = params["train"]["metric"]
n_neighbors = params["train"]["neighbors"]

# Filter active users/products
user_counts = df['UserId'].value_counts()
product_counts = df['ProductId'].value_counts()
active_users = user_counts[user_counts >= min_user_interactions].index
active_products = product_counts[product_counts >= min_product_interactions].index
df = df[df['UserId'].isin(active_users) & df['ProductId'].isin(active_products)]

# Create user-item matrix
user_product_matrix = df.pivot_table(index='UserId', columns='ProductId', values='Score').fillna(0)
sparse_matrix = csr_matrix(user_product_matrix.values)

# Train KNN model
model = NearestNeighbors(metric=metric, algorithm='brute', n_neighbors=n_neighbors)
model.fit(sparse_matrix)

# Save model and metadata
with open(output_model, 'wb') as f:
    pickle.dump((model, user_product_matrix), f)