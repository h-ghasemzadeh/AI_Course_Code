import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Load the uploaded file
file_path = "family_words.csv"
data = pd.read_csv(file_path)

# Extract the words from the file
words = data['Words_Family']

# Vectorize the words using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
X = vectorizer.fit_transform(words)

# Print the dimensions of X
print("The dimensions of X are:", X.shape)
# Print the dense representation of X
print("The vector X (TF-IDF matrix) is:\n", X.toarray())

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X.toarray())

# Add jittering for better readability
def add_jitter(points, scale=0.2):
    jitter = np.random.normal(0, scale, points.shape)
    return points + jitter

# Apply jitter to points for visualization
jittered_points = add_jitter(reduced_X, scale=0.2)

# Plot words before clustering
plt.figure(figsize=(12, 8))
plt.scatter(jittered_points[:, 0], jittered_points[:, 1], c='gray', edgecolor='k')
for i, word in enumerate(words):
    plt.text(jittered_points[i, 0], jittered_points[i, 1], word, fontsize=8)
plt.title('Words Before Clustering (Unclustered)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.grid(True)
plt.show()

# Apply K-means clustering
n_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Calculate Silhouette Score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Assign clusters to words
data['Cluster'] = kmeans.labels_

# Plot words after clustering
plt.figure(figsize=(12, 8))
colors = plt.colormaps['tab10']  # Updated colormap syntax

for cluster in range(n_clusters):
    cluster_indices = (data['Cluster'] == cluster).values
    cluster_points = jittered_points[cluster_indices]
    cluster_words = words[cluster_indices]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}', c=[colors(cluster)])
    for i, word in enumerate(cluster_words):
        plt.text(cluster_points[i, 0], cluster_points[i, 1], f"{word} ({cluster + 1})", fontsize=8)

plt.title('Clustered Words with Numbers and Colors')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Save the clustered data for further use
clustered_file_path = "family_words_clustered.csv"
data.to_csv(clustered_file_path, index=False)

# Classify new words
new_words = ["Acting", "Runaway", "Builder", "Speakerman", "Piano"]  # Example new words
new_X = vectorizer.transform(new_words)  # Vectorize the new words using the same vectorizer
new_clusters = kmeans.predict(new_X)  # Predict clusters for new words

# Print the cluster for each new word
print("\nCluster assignments for new words:")
for word, cluster in zip(new_words, new_clusters):
    print(f"'{word}' belongs to Cluster {cluster + 1}")
