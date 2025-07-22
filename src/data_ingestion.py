import kaggle
import pandas as pd
import numpy as np
import os
import sys

def download_data(dataset_name: str, path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)
        return os.path.join(path, 'Reviews.csv')
    except kaggle.api.kaggle_api_extended.KaggleError as e:
        print(f"Error downloading dataset from Kaggle: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Kaggle API credentials not found. Please configure kaggle.json")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating directory or downloading data: {e}")
        sys.exit(1)

def read_data(data_path: str = 'data/raw/Reviews.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Data file {data_path} is empty")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading data from {data_path}: {e}")
        sys.exit(1)

def main() -> None:
    try:
        dataset_name = 'snap/amazon-fine-food-reviews'
        path = 'data/raw'
        data_path = download_data(dataset_name, path)
        df = read_data(data_path)
        
        # Save the raw data to a CSV file
        output_file = 'data/raw/Reviews.csv'
        df.to_csv(output_file, index=False)
        print(f"Data ingestion completed. Data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error in data ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()