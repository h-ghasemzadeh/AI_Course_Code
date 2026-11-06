import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)

# Generate data (age, exercise level between 0 and 10) for three classes: healthy, cancerous, and unknown
healthy = np.random.multivariate_normal([45, 5], [[40, 3], [1, 3]], 80)
cancerous = np.random.multivariate_normal([60, 3], [[50, .4], [.5, 3]], 30)
unknown = np.random.multivariate_normal([55, 3], [[35, .5], [0.3, 1]], 30)

# Clip exercise level to ensure values remain between 0 and 10
healthy[:, 1] = np.clip(healthy[:, 1], 0, 8)
cancerous[:, 1] = np.clip(cancerous[:, 1], 0, 8)
unknown[:, 1] = np.clip(unknown[:, 1], 0, 8)

# Clip age to ensure values remain between 0 and 100
healthy[:, 0] = np.clip(healthy[:, 0], 0, 100)
cancerous[:, 0] = np.clip(cancerous[:, 0], 0, 100)
unknown[:, 0] = np.clip(unknown[:, 0], 0, 100)

# Data labels
X = np.vstack([healthy, cancerous, unknown])
y = np.array([0]*80 + [1]*30 + [2]*30)

# Apply additional weight to the exercise feature (column 1)
exercise_weight = 2.0  # Adjust this weight as needed
X[:, 1] *= exercise_weight

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set up color maps for visualization
cmap_light = ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cmap_bold = ['g', 'r', 'b']

# Function to plot decision boundaries
def plot_decision_boundaries(k):
    # Train k-NN model with distance-based weighting
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X, y)

    # Generate a mesh of points to plot decision boundaries
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = (0 * exercise_weight), (10 * exercise_weight)  # Adjust for exercise weight
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict labels for each point on the mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contour and data points
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k', s=30)
    plt.xlabel("Age")
    plt.ylabel("Exercise Level (Weighted)")
    plt.title(f"Decision Boundaries for k = {k}")
    plt.ylim(0 * exercise_weight, 8 * exercise_weight)  # Limit y-axis to [0, weighted max]
    plt.show()

# Plot decision boundaries for different values of k
k_values = [1, 3, 5, 7, 13, 15, 20, 25]
for k in k_values:
    plot_decision_boundaries(k)

# Evaluate each k value
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for k={k}: {accuracy:.2f}")

# Plot accuracy vs. k value
plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("k-NN Accuracy for Different k Values")
plt.show()

# Find and print the best k
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\nBest k value: {best_k} with accuracy of {best_accuracy:.2f}")
