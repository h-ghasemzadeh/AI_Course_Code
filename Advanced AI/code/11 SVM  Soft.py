import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Generate a 2D dataset for visualization
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to plot decision boundary, margin, and support vectors
def plot_svm_with_margin(X, y, model, kernel_name, accuracy):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm, s=20)
    plt.scatter(
        model.support_vectors_[:, 0], model.support_vectors_[:, 1],
        s=100, facecolors='none', edgecolors='k', label='Support Vectors'
    )
    plt.title(f"{kernel_name} Kernel SVM (Accuracy: {accuracy:.2f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Function to train, evaluate, and plot SVM for different kernels
def train_evaluate_plot(kernel, C=1.0, degree=3, gamma='scale', coef0=0.0):
    print(f"Kernel: {kernel}, C: {C}, Degree: {degree}, Gamma: {gamma}, Coef0 (r): {coef0}")
    svm = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    plot_svm_with_margin(X_train, y_train, svm, kernel.capitalize(), accuracy)
    ConfusionMatrixDisplay.from_estimator(svm, X_test, y_test)
    plt.show()

# Test the function with different kernels
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    train_evaluate_plot(kernel, C=1.0, degree=3, gamma='scale', coef0=0.0)
