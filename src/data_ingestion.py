import kaggle
import pandas as pd
import numpy as np
import os

# Ensure the raw data directory exists
os.makedirs('data/raw', exist_ok=True)

# Use Kaggle API to download the dataset
kaggle.api.dataset_download_files('snap/amazon-fine-food-reviews', path='data/raw', unzip=True)

# Load the dataset into a DataFrame
df = pd.read_csv('data/raw/Reviews.csv')