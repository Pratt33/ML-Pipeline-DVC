import pandas as pd
import os

input_file = "data/raw/Reviews.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "processed_reviews.csv")

df = pd.read_csv(input_file)


# Drop unnecessary columns
keep_cols = ['Id', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']
df = df[keep_cols]

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

df['Summary'] = df['Summary'].str.lower().str.strip()
df['Text'] = df['Text'].str.lower().str.strip()

df.to_csv(output_file, index=False)