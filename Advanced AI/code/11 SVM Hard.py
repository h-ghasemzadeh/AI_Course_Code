import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Generate synthetic linearly separable data
def generate_data():
    """
    Generates a linearly separable dataset with 50 samples.
    """
    X, y = make_blobs(n_samples=50, centers=2, cluster_std=1.0, random_state=42)
    y = 2 * y - 1  # Convert labels to {-1, 1} for SVM
    return X, y

# Step 2: Plot decision boundaries with margins
def plot_svm_with_margin(X, y, model):
    """
    Plot the decision boundary, margins, and support vectors for a 2D dataset.
    """
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Predict the class for each grid point
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    
    # Highlight support vectors
    plt.scatter(
        model.support_vectors_[:, 0], model.support_vectors_[:, 1],
        s=100, facecolors='none', edgecolors='k', label='Support Vectors'
    )
    
    # Plot margins
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    margin = 1 / np.linalg.norm(coef)
    
    # Equation of the decision boundary: coef[0]*x + coef[1]*y + intercept = 0
    slope = -coef[0] / coef[1]
    intercept_decision = -intercept / coef[1]
    intercept_margin_up = intercept_decision + margin / coef[1]
    intercept_margin_down = intercept_decision - margin / coef[1]
    
    x_vals = np.linspace(x_min, x_max, 100)
    plt.plot(x_vals, slope * x_vals + intercept_decision, 'k-', label='Decision Boundary')
    plt.plot(x_vals, slope * x_vals + intercept_margin_up, 'k--', label='Margin +1')
    plt.plot(x_vals, slope * x_vals + intercept_margin_down, 'k--', label='Margin -1')
    
    plt.title(f"Hard-Margin SVM Decision Boundary (Margin: {margin:.2f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Step 3: Train and evaluate the SVM
def train_and_evaluate_hard_margin():
    """
    Train and evaluate a hard-margin SVM on synthetic data.
    """
    # Generate data
    X, y = generate_data()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a hard-margin SVM (set C to a very large value)
    hard_svm = SVC(C=1e6, kernel='linear')
    hard_svm.fit(X_scaled, y)
    
    # Extract and print the normal vector
    coef = hard_svm.coef_[0]
    intercept = hard_svm.intercept_[0]
    print(f"Normal vector (\\(\\beta\\)): {coef}")
    print(f"Formula of the decision boundary: {coef[0]:.2f} * x1 + {coef[1]:.2f} * x2 + {intercept:.2f} = 0")
    
    # Predict on the training set
    y_pred = hard_svm.predict(X_scaled)
    
    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    
    # Plot decision boundary and margins
    plot_svm_with_margin(X_scaled, y, hard_svm)

# Step 4: Run the function
train_and_evaluate_hard_margin()
