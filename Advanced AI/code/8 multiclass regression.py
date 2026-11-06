import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
# Load dataset (e.g., iris dataset for multiclass classification)
data = load_iris() # returns a dictionary-like object with the data and metadata about the dataset.
X = data.data      # extracts the feature data (or inputs:Sepal length,Sepal width,Petal length)
y = data.target    # provides the target labels (or outputs) for each iris flower (Setosa,Versicolour,Virginica)
print("Length of data" , len(X))
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Initialize logistic regression model for multiclass
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
logreg.fit(X_train, y_train)
# Predict and evaluate
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Use PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Plot True and Predicted Labels
plt.figure(figsize=(10, 5))

# Plot True Labels
plt.subplot(1, 2, 1)
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=50)
plt.title("True Labels")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Plot Predicted Labels
plt.subplot(1, 2, 2)
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
plt.title("Predicted Labels")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.show()

# Decision Boundary Plot
# Define mesh for plot
x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on mesh grid
Z = logreg.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, edgecolor='k', cmap='viridis', s=50)
plt.title("Decision Boundary with True Labels")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
