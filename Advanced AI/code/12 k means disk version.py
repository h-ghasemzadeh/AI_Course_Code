import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Load the data
file_path = "family_words.csv"
data = pd.read_csv(file_path)

# Extract words
words = data['Words_Family']

# Vectorize the words using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
X = vectorizer.fit_transform(words)

# Apply Mini-Batch K-Means
n_clusters = 5  # Number of clusters
mini_batch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10, random_state=42)
mini_batch_kmeans.fit(X)

# Assign clusters
data['Cluster'] = mini_batch_kmeans.labels_

# Calculate Silhouette Score
silhouette_avg = silhouette_score(X, mini_batch_kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Save clustered data
clustered_file_path = "family_words_mini_batch_clustered.csv"
data.to_csv(clustered_file_path, index=False)

# Print clusters for reference
print("\nCluster assignments:")
print(data)
