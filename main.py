import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- LOAD DATASET ----
# Update the path to your dataset
data = pd.read_csv("sentimentdataset.csv")

# ---- BASIC DATA INSPECTION ----
print("Dataset Preview:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nMissing Values per Column:")
print(data.isnull().sum())

# ---- DATA CLEANING ----
def clean_text(text):
    text = str(text).lower()  # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['clean_text'] = data['Text'].apply(clean_text)

# ---- HANDLE MISSING VALUES ----
data.dropna(subset=['clean_text', 'Sentiment'], inplace=True)

# ---- EXPLORATORY DATA ANALYSIS ----
# Limit categories to top 10 most frequent
top_sentiments = data['Sentiment'].value_counts().nlargest(10).index
filtered_data = data[data['Sentiment'].isin(top_sentiments)]

plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', hue='Sentiment', data=filtered_data, palette='viridis', legend=False)
plt.title('Top 10 Sentiment Categories')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Word Cloud for Sentiments
all_words = ' '.join(filtered_data['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Top 10 Sentiments")
plt.show()

# ---- TREND ANALYSIS OVER TIME ----
if 'Year' in data.columns:
    yearly_sentiment = (
        filtered_data.groupby(['Year', 'Sentiment'])
        .size()
        .unstack(fill_value=0)
    )
    # Keep only top 10 columns (sentiments)
    yearly_sentiment = yearly_sentiment[top_sentiments.intersection(yearly_sentiment.columns)]
    yearly_sentiment.plot(kind='line', figsize=(10, 6))
    plt.title("Sentiment Trends Over the Years (Top 10 Sentiments)")
    plt.xlabel("Year")
    plt.ylabel("Number of Posts")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ---- FEATURE EXTRACTION ----
X = data['clean_text']
y = data['Sentiment']

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

# ---- TRAIN-TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# ---- MODEL TRAINING ----
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ---- MODEL EVALUATION ----
y_pred = model.predict(X_test)

print("\n--- MODEL EVALUATION ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---- HASHTAG TREND ANALYSIS ----
if 'Hashtags' in data.columns:
    all_hashtags = data['Hashtags'].dropna().astype(str).str.lower().str.replace('[', '').str.replace(']', '')
    hashtags_list = []
    for tags in all_hashtags:
        hashtags_list.extend([tag.strip().replace("'", "") for tag in tags.split(',') if tag.strip()])
    hashtags_series = pd.Series(hashtags_list)
    top_hashtags = hashtags_series.value_counts().head(10)
    top_hashtags.plot(kind='bar', color='teal', figsize=(8, 5))
    plt.title("Top 10 Trending Hashtags")
    plt.xlabel("Hashtag")
    plt.ylabel("Frequency")
    plt.show()

# ---- SAVE MODEL OUTPUTS ----
results = pd.DataFrame({'Text': X, 'Predicted Sentiment': model.predict(X_vec)})
results.to_csv('sentiment_predictions.csv', index=False)
print("\nResults saved to 'sentiment_predictions.csv'")

print("\n✅ Sentiment Analysis & Trend Model Completed Successfully!")
