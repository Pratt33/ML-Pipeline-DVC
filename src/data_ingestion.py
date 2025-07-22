import kaggle
import pandas as pd
import numpy as np
import os
import sys
import logging

#logging configuration
#logging in logging.getLogger is used to avoid circular imports means that the logger is not imported from the main module it is defined in this module
#while getLogger is used to get the logger instance this instance is used to log messages in this module changing the instance name will not affect the main module it is just a convention to use the module name as the logger name
logger=logging.getLogger('data_ingestion')
#set level is used to set the logging level to INFO this means that only messages with level INFO or higher will be logged
#this is useful to avoid logging too much information in production environments
#so in prod envs we can set the level to WARNING or ERROR to log only important messages only
#in development environments we can set the level to DEBUG to log all messages
logger.setLevel(logging.INFO)

#console_handler is used to log messages to the console this is useful for debugging and development purposes
#we can also add a file handler to log messages to a file for production purposes
#along with file there is console handler to log messages to the console
#stream_handler is used to log messages to the console this is useful for debugging and development purposes
#for file_handler we can use FileHandler to log messages to a file
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Set console handler to DEBUG level

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def download_data(dataset_name: str, path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)
        return os.path.join(path, 'Reviews.csv')
    except kaggle.api.kaggle_api_extended.KaggleError as e:
        #print(f"Error downloading dataset from Kaggle: {e}")
        logger.error(f"Error downloading dataset from Kaggle: {e}")
        sys.exit(1)
    except FileNotFoundError:
        #print("Error: Kaggle API credentials not found. Please configure kaggle.json")
        logger.error("Error: Kaggle API credentials not found. Please configure kaggle.json")
        sys.exit(1)
    except Exception as e:
        #print(f"Error creating directory or downloading data: {e}")
        logger.error(f"Error creating directory or downloading data: {e}")
        sys.exit(1)

def read_data(data_path: str = 'data/raw/Reviews.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        #print(f"Error: Data file not found at {data_path}")
        logger.error(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Error: Data file {data_path} is empty")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
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
        #print(f"Error in data ingestion: {e}")
        logger.debug(f"Error in data ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()