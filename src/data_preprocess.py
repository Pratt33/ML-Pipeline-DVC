import pandas as pd
import os
import sys

def paths():
    input_file = "data/raw/Reviews.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_reviews.csv")
    return input_file, output_file

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
        print(f"Error: Missing required column {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

def save_data(df: pd.DataFrame, output_file: str) -> str:
    try:
        df.to_csv(output_file, index=False)
        return output_file
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")
        sys.exit(1)

def main():
    try:
        # Process the data
        input_file, output_file = paths()
        df = pd.read_csv(input_file)
        df = process_data(df)
        output_file = save_data(df, output_file)
        print(f"Data preprocessing completed. Output saved to: {output_file}")
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()