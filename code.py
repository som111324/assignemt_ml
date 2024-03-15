import pandas as pd

# Load the dataset
df = pd.read_csv('data.csv')

# Display the first few rows of the dataset
print(df.head())

# Check the column names and data types
print(df.info())

# Explore the distribution of different categories of financial news articles
print(df['category'].value_counts())




import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

# Load the CSV file
df = pd.read_csv('data.csv')

# Display the column names and first few rows
print(df.columns)
print(df.head())

# Check if 'text' column exists or use an alternative column for text preprocessing
if 'text' in df.columns:
    text_column = 'text'
elif 'content' in df.columns:
    text_column = 'content'
else:
    raise KeyError("No suitable text column found in the DataFrame.")

# Preprocess text data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df[text_column].apply(preprocess_text)

# Display the preprocessed text column
print(df['clean_text'].head())


from gensim.summarization import summarize

# Function to generate summaries
def generate_summary(text):
    return summarize(text, ratio=0.2)  # Adjust ratio for desired summary length

# Apply summarization to the 'clean_text' column
df['summary'] = df['clean_text'].apply(generate_summary)

# Display the original text and its corresponding summary
for index, row in df.iterrows():
    print(f"Original Text ({index + 1}):\n{row['clean_text']}\n")
    print(f"Summary ({index + 1}):\n{row['summary']}\n")
    print('-' * 50)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming df['summary'] contains the summaries after text summarization

# Choose a sample article (change index as needed)
sample_article_index = 0
sample_summary = df.loc[sample_article_index, 'summary']

# Calculate TF-IDF vectors for summaries
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['summary'])

# Calculate cosine similarity between sample summary and all summaries
cosine_similarities = cosine_similarity(tfidf_vectorizer.transform([sample_summary]), tfidf_matrix)
similarity_scores = list(enumerate(cosine_similarities[0]))

# Set a similarity threshold (adjust as needed)
similarity_threshold = 0.8

# Retrieve similar articles
similar_articles = [(index, score) for index, score in similarity_scores if score >= similarity_threshold]

# Display similar articles
print(f"Sample Article:\n{sample_summary}\n")
print("Similar Articles:")
for article_index, similarity_score in similar_articles:
    print(f'Similarity Score: {similarity_score}')
    print(df.loc[article_index, 'summary'])
    print('-' * 50)






#keyword Analysis
from collections import Counter

# Concatenate all summaries into a single string
all_summaries = ' '.join(df['summary'])

# Tokenize the text and count frequencies of words
words = all_summaries.split()
word_counts = Counter(words)

# Print the most common keywords
print("Most Common Keywords:")
print(word_counts.most_common(10))  # Adjust the number as needed

#clustering 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['summary'])

# Perform K-means clustering
num_clusters = 3  # Adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Print clusters and sample articles from each cluster
for cluster_id in range(num_clusters):
    cluster_articles = df[df['cluster'] == cluster_id]['summary'].tolist()
    print(f"Cluster {cluster_id}:")
    for article in cluster_articles[:3]:  # Print first 3 articles in each cluster
        print(article)
    print('-' * 50)
