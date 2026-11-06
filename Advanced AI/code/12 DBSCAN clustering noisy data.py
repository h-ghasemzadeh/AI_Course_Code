import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the rings data from the CSV file
csv_file_path = "two_rings_data.csv"
data = pd.read_csv(csv_file_path)

# Extract the coordinates
X = data[['x', 'y']]

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust `eps` and `min_samples` as needed
data['Cluster'] = dbscan.fit_predict(X)

# Plot the clustered data
plt.figure(figsize=(8, 8))
unique_clusters = set(data['Cluster'])

# Visualize clusters
for cluster in unique_clusters:
    cluster_data = data[data['Cluster'] == cluster]
    label = f'Cluster {cluster}' if cluster != -1 else 'Noise'
    plt.scatter(cluster_data['x'], cluster_data['y'], label=label)

plt.title("DBSCAN Clustering of Rings")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Print cluster assignments
print(data.head())
