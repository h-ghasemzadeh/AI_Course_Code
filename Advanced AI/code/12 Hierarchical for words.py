import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
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

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X.toarray())

# Generate linkage matrix for hierarchical clustering
linkage_matrix = linkage(X.toarray(), method='ward')

# Adjust the dendrogram plotting for better scaling and readability
plt.figure(figsize=(16, 12))  # Larger figure size
dendrogram(
    linkage_matrix,
    labels=words.tolist(),
    leaf_rotation=90,  # Rotate labels for better visibility
    leaf_font_size=10,  # Increase font size for readability
    color_threshold=0.7 * max(linkage_matrix[:, 2]),  # Color branches below threshold
)

plt.title('Dendrogram for Hierarchical Clustering', fontsize=16)
plt.xlabel('Words', fontsize=14)
plt.ylabel('Distance', fontsize=14)
plt.tight_layout()  # Ensure labels and plot fit within the figure

# Adjust layout to prevent label truncation
plt.subplots_adjust(bottom=0.3, left=0.1, right=0.9, top=0.9)  # Adjust margins: bottom, left, right, top

plt.show()

# Cut the dendrogram into clusters
n_clusters = 5  # Adjust the number of clusters as needed
clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Assign clusters to the data
data['Cluster'] = clusters

# Compute the silhouette score for cluster evaluation
silhouette_avg = silhouette_score(X, clusters, metric='euclidean')
print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}")

# Create a silhouette plot for detailed analysis
silhouette_vals = silhouette_samples(X, clusters, metric='euclidean')

plt.figure(figsize=(10, 8))
y_lower = 10
for i in range(1, n_clusters + 1):
    ith_cluster_silhouette_vals = silhouette_vals[clusters == i]
    ith_cluster_silhouette_vals.sort()
    size_cluster_i = ith_cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.tab10(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}', fontsize=10)
    y_lower = y_upper + 10  # Add space between clusters

plt.title('Silhouette Plot for Hierarchical Clustering', fontsize=16)
plt.xlabel('Silhouette Coefficient Values', fontsize=14)
plt.ylabel('Cluster Label', fontsize=14)
plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Average Silhouette Score ({silhouette_avg:.2f})')
plt.legend()
plt.tight_layout()
plt.show()

# Plot words after clustering using PCA for visualization
plt.figure(figsize=(12, 8))
colors = plt.get_cmap('tab10')

for cluster in range(1, n_clusters + 1):
    cluster_indices = (data['Cluster'] == cluster).values
    cluster_points = reduced_X[cluster_indices]
    cluster_words = words[cluster_indices]
    plt.scatter(
        cluster_points[:, 0], 
        cluster_points[:, 1], 
        label=f'Cluster {cluster}', 
        c=[colors(cluster / n_clusters)]
    )
    for i, word in enumerate(cluster_words):
        plt.text(
            cluster_points[i, 0], 
            cluster_points[i, 1], 
            f"{word} ({cluster})", 
            fontsize=8
        )

plt.title('Clustered Words (Hierarchical Clustering)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Save the clustered data for further use
clustered_file_path = "family_words_hierarchical_clustered.csv"
data.to_csv(clustered_file_path, index=False)

# Print the cluster assignments
print("\nCluster assignments:")
print(data[['Words_Family', 'Cluster']])

# Verification: Display the contents of each cluster
for cluster in range(1, n_clusters + 1):
    cluster_words = data[data['Cluster'] == cluster]['Words_Family'].tolist()
    print(f"\nCluster {cluster} contains {len(cluster_words)} words:")
    print(cluster_words)

# Optional: Check for clusters with very few members
for cluster in range(1, n_clusters + 1):
    cluster_size = len(data[data['Cluster'] == cluster])
    if cluster_size < 2:
        print(f"Warning: Cluster {cluster} has very few members ({cluster_size} words). Consider reviewing.")
