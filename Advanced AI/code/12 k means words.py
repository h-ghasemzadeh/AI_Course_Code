import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the uploaded file
file_path = "family_words.csv"
data = pd.read_csv(file_path)

# Extract the words from the file (correcting the column name)
words = data['Words_Family']

# Vectorize the words using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
X = vectorizer.fit_transform(words)

# Apply K-means clustering
n_clusters = 5  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Assign clusters to words
data['Cluster'] = kmeans.labels_

# Visualize the clustering results
for cluster in range(n_clusters):
    cluster_words = data[data['Cluster'] == cluster]['Words_Family'].values
    print(f"Cluster {cluster + 1}:")
    print(", ".join(cluster_words))
    print()

# Save the clustered data for further use
clustered_file_path = "family_words_clustered.csv"
data.to_csv(clustered_file_path, index=False)
