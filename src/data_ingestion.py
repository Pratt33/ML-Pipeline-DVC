import kaggle
import pandas as pd
import numpy as np
import os

def download_data(dataset_name, path):
    kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)
    return os.path.join(path, 'Reviews.csv')

def read_data(data_path='data/raw/Reviews.csv'):
    # Correct the file path to match the actual location
    df = pd.read_csv(data_path)
    return df

def main():
    dataset_name = 'snap/amazon-fine-food-reviews'
    path= 'data/raw'
    data_path = download_data(dataset_name, path)
    df = read_data(data_path)
    
    # Save the raw data to a CSV file
    output_file = 'data/raw/Reviews.csv'
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()