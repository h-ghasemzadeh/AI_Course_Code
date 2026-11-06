import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
# Optional: Disable oneDNN custom operations to avoid floating-point warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Generate a 2D dataset for visualization
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the ANN model
def build_model(input_dim, output_dim):
    """
    Build a simple feedforward ANN model with two hidden layers.
    """
    model = Sequential([
        Input(shape=(input_dim,)),  # Explicit Input layer
        Dense(4, activation='relu'),  # 8 neuron First hidden layer
        Dense(2, activation='relu'),  # 4 neuron Second hidden layer

        Dense(output_dim, activation='sigmoid')  # Output layer for binary classification
    ])
    return model

# Plot decision boundary
def plot_ann_with_boundary(X, y, model, accuracy):
    """
    Plot the decision boundary for the trained ANN model.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm, s=20)
    plt.title(f"ANN Decision Boundary (Accuracy: {accuracy:.2f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Train, evaluate, and plot ANN
def train_evaluate_ann(epochs=50, batch_size=8):
    """
    Train the ANN, evaluate on test data, and visualize results.
    """
    print("Training Artificial Neural Network...")
    
    # Build and compile the model
    model = build_model(input_dim=2, output_dim=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Evaluate on test set
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    # Plot decision boundary
    plot_ann_with_boundary(X_train, y_train, model, accuracy)
    
    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.show()
    
    # Plot training loss and accuracy
    plot_training_history(history)

# Function to plot training history
def plot_training_history(history):
    """
    Plot the training loss and accuracy over epochs.
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the ANN training and visualization
if __name__ == "__main__":
    try:
        train_evaluate_ann(epochs=50, batch_size=8)
    except Exception as e:
        print(f"An error occurred: {e}")
