import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the rings data from the CSV file
csv_file_path = "two_rings_data.csv"
data = pd.read_csv(csv_file_path)

# Extract the coordinates
X = data[['x', 'y']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Plot the clustered data
plt.figure(figsize=(8, 8))
for cluster in range(2):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster + 1}')

# Mark centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title("K-Means Clustering of Rings")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Print cluster assignments
print(data.head())
