import pandas as pd
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Load the rings data from the CSV file
csv_file_path = "two_rings_data.csv"
data = pd.read_csv(csv_file_path)

# Extract the coordinates
X = data[['x', 'y']]

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
data['Cluster'] = spectral.fit_predict(X)

# Plot the clustered data
plt.figure(figsize=(8, 8))
for cluster in range(2):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster + 1}')

plt.title("Spectral Clustering of Rings")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Print cluster assignments
print(data.head())
