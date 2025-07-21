import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import os

# Load data
df = pd.read_csv('data/processed/processed_reviews.csv')

os.makedirs('visualizations', exist_ok=True)

# 1. Distribution of review scores
plt.figure(figsize=(8,5))
sns.countplot(x='Score', data=df, palette='viridis')
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.savefig('visualizations/score_distribution.png')
plt.close()

# 2. Top 10 most reviewed products
top_products = df['ProductId'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_products.values, y=top_products.index, palette='mako')
plt.title('Top 10 Most Reviewed Products')
plt.xlabel('Number of Reviews')
plt.ylabel('Product ID')
plt.savefig('visualizations/top_products.png')
plt.close()

# 3. Top 10 most active users
top_users = df['UserId'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_users.values, y=top_users.index, palette='crest')
plt.title('Top 10 Most Active Users')
plt.xlabel('Number of Reviews')
plt.ylabel('User ID')
plt.savefig('visualizations/top_users.png')
plt.close()

# 4. Distribution of review lengths
review_lengths = df['Text'].str.split().apply(len)
plt.figure(figsize=(8,5))
sns.histplot(review_lengths, bins=50, kde=True, color='skyblue')
plt.title('Distribution of Review Lengths (words)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.savefig('visualizations/review_length_distribution.png')
plt.close()

# 5. Word cloud of review summaries
text = ' '.join(df['Summary'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Review Summaries')
plt.savefig('visualizations/summary_wordcloud.png')
plt.close()

# 6. Correlation between score and review length
plt.figure(figsize=(8,5))
sns.boxplot(x='Score', y=review_lengths, data=df, palette='Set2')
plt.title('Review Length by Score')
plt.xlabel('Score')
plt.ylabel('Review Length (words)')
plt.savefig('visualizations/score_vs_length.png')
plt.close()

print('All visualizations saved in the visualizations/ directory.') 