import kaggle
import pandas as pd
import numpy as np
import os

kaggle.api.dataset_download_files('snap/amazon-fine-food-reviews', path='data/raw', unzip=True)

# Correct the file path to match the actual location
df=pd.read_csv('data/raw/Reviews.csv')