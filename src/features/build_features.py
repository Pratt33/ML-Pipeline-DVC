import pandas as pd
import os
import sys
import logging
import numpy as np
from scipy.sparse import csr_matrix
import yaml

# Set up logging
logger = logging.getLogger('data_preprocess')
logger.setLevel(logging.INFO)
console_handler = logging.FileHandler('data_preprocess.log')
console_handler.setLevel(logging.ERROR)  # Set console handler to ERROR level

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def paths():
    try:
        input_file = "data/raw/Reviews.csv"
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "processed_reviews.csv")
        return input_file, output_file
    except Exception as e:
        logger.error(f"Error in paths function: {e}")
        sys.exit(1)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Drop unnecessary columns
        keep_cols = ['Id', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']
        df = df[keep_cols].copy()

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df.loc[:, 'Summary'] = df['Summary'].str.lower().str.strip()
        df.loc[:, 'Text'] = df['Text'].str.lower().str.strip()

        return df
    except KeyError as e:
        logger.error(f"Error: Missing required column {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        sys.exit(1)

def save_data(df: pd.DataFrame, output_file: str) -> str:
    try:
        df.to_csv(output_file, index=False)
        return output_file
    except Exception as e:
        logger.error(f"Error saving data to {output_file}: {e}")
        sys.exit(1)

def create_user_item_matrix(df: pd.DataFrame, min_interactions: int = 5):
    """
    Create user-item interaction matrix from reviews data
    """
    logger.info("Creating user-item matrix...")
    
    # Filter users and items with minimum interactions
    user_counts = df['reviewerID'].value_counts()
    item_counts = df['asin'].value_counts()
    
    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index
    
    df_filtered = df[
        (df['reviewerID'].isin(valid_users)) & 
        (df['asin'].isin(valid_items))
    ].copy()
    
    logger.info(f"Filtered data: {len(df_filtered)} interactions, {len(valid_users)} users, {len(valid_items)} items")
    
    # Create mappings
    unique_users = df_filtered['reviewerID'].unique()
    unique_items = df_filtered['asin'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create user-item matrix
    user_indices = df_filtered['reviewerID'].map(user_to_idx)
    item_indices = df_filtered['asin'].map(item_to_idx)
    ratings = df_filtered['overall'].values
    
    matrix = csr_matrix(
        (ratings, (user_indices, item_indices)),
        shape=(len(unique_users), len(unique_items))
    )
    
    return matrix, user_to_idx, item_to_idx, df_filtered

def main():
    try:
        # Load parameters
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        # Load raw data
        logger.info("Loading raw data...")
        df = pd.read_csv('data/raw/Reviews.csv')
        
        # Create user-item matrix
        matrix, user_mapping, item_mapping, filtered_df = create_user_item_matrix(
            df, min_interactions=params['evaluate']['min_interactions']
        )
        
        # Convert sparse matrix to dense for saving
        matrix_dense = matrix.toarray()
        
        # Create directories
        os.makedirs('data/processed', exist_ok=True)
        
        # Save user-item matrix
        matrix_df = pd.DataFrame(matrix_dense)
        matrix_df.to_csv('data/processed/user_item_matrix.csv', index=False)
        
        # Save mappings
        user_mapping_df = pd.DataFrame(list(user_mapping.items()), columns=['user_id', 'user_idx'])
        item_mapping_df = pd.DataFrame(list(item_mapping.items()), columns=['item_id', 'item_idx'])
        
        user_mapping_df.to_csv('data/processed/user_mapping.csv', index=False)
        item_mapping_df.to_csv('data/processed/item_mapping.csv', index=False)
        
        logger.info(f"User-item matrix shape: {matrix.shape}")
        logger.info(f"Matrix sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")
        logger.info("Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    main()