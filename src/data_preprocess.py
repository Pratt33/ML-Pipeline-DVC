import pandas as pd
import os
def paths():
    input_file = "data/raw/Reviews.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_reviews.csv")

    return input_file, output_file

# Read the data from the input file

def process_data(df):
    # Drop unnecessary columns
    keep_cols = ['Id', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']
    df = df[keep_cols].copy()

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df.loc[:, 'Summary'] = df['Summary'].str.lower().str.strip()
    df.loc[:, 'Text'] = df['Text'].str.lower().str.strip()

    return df
def save_data(df, output_file):
    df.to_csv(output_file, index=False)
    return output_file

def main():
    # Process the data
    df = pd.read_csv("data/raw/Reviews.csv")
    input_file, output_file = paths()
    df = process_data(df)
    output_file=save_data(df, output_file)


if __name__ == "__main__":
    main()