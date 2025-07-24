import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import yaml
import logging
import os
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_user_item_matrix(df: pd.DataFrame, min_interactions: int = 5):
    """
    Create user-item interaction matrix from reviews data
    """
    logger.info("Creating user-item matrix...")
    
    # Filter users and items with minimum interactions
    user_counts = df['UserId'].value_counts()
    item_counts = df['ProductId'].value_counts()
    
    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index
    
    df_filtered = df[
        (df['UserId'].isin(valid_users)) & 
        (df['ProductId'].isin(valid_items))
    ].copy()
    
    logger.info(f"Filtered data: {len(df_filtered)} interactions, {len(valid_users)} users, {len(valid_items)} items")
    
    # Create mappings
    unique_users = df_filtered['UserId'].unique()
    unique_items = df_filtered['ProductId'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create user-item matrix (SPARSE!)
    user_indices = df_filtered['UserId'].map(user_to_idx)
    item_indices = df_filtered['ProductId'].map(item_to_idx)
    ratings = df_filtered['Score'].values
    
    matrix = csr_matrix(
        (ratings, (user_indices, item_indices)),
        shape=(len(unique_users), len(unique_items))
    )
    
    return matrix, user_to_idx, item_to_idx, df_filtered

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    try:
        # Load raw data
        logger.info("Loading raw data...")
        df = pd.read_csv('data/raw/Reviews.csv')
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Create user-item matrix
        matrix, user_mapping, item_mapping, filtered_df = create_user_item_matrix(
            df, min_interactions=params['train']['min_user_interactions']
        )
        
        # Create directories
        os.makedirs('data/processed', exist_ok=True)
        
        # Save SPARSE matrix (efficient!)
        sparse.save_npz('data/processed/user_item_matrix.npz', matrix)
        
        # Save matrix metadata
        matrix_info = {
            'shape': matrix.shape,
            'nnz': matrix.nnz,
            'sparsity': 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
            'dtype': str(matrix.dtype)
        }
        
        with open('data/processed/matrix_info.pkl', 'wb') as f:
            pickle.dump(matrix_info, f)
        
        # Save mappings
        user_mapping_df = pd.DataFrame(list(user_mapping.items()), columns=['user_id', 'user_idx'])
        item_mapping_df = pd.DataFrame(list(item_mapping.items()), columns=['item_id', 'item_idx'])
        
        user_mapping_df.to_csv('data/processed/user_mapping.csv', index=False)
        item_mapping_df.to_csv('data/processed/item_mapping.csv', index=False)
        
        # Log statistics
        total_cells = matrix.shape[0] * matrix.shape[1]
        memory_dense_mb = total_cells * 8 / (1024**2)  # 8 bytes per float64
        memory_sparse_mb = matrix.data.nbytes / (1024**2)
        
        logger.info(f"User-item matrix shape: {matrix.shape}")
        logger.info(f"Total possible cells: {total_cells:,}")
        logger.info(f"Non-zero cells: {matrix.nnz:,}")
        logger.info(f"Sparsity: {matrix_info['sparsity']*100:.2f}%")
        logger.info(f"Memory - Dense: {memory_dense_mb:.1f} MB")
        logger.info(f"Memory - Sparse: {memory_sparse_mb:.1f} MB")
        logger.info(f"Space savings: {memory_dense_mb/memory_sparse_mb:.0f}x smaller!")
        logger.info("Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    main()