# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
 
# Load and preprocess the Iris dataset
def load_and_preprocess_data():
    """
    Load the Iris dataset, normalize features, and one-hot encode target labels.
    Splits the data into training and test sets.
    """
    # Load the dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target labels

    # One-hot encode the target labels
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)

    # Normalize the input features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, iris.target_names, scaler

# Build the ANN model
def build_model(input_dim, output_dim):
    """
    Build a simple feedforward ANN model with two hidden layers.
    """
    model = Sequential([
        Dense(8, input_dim=input_dim, activation='relu'),  # First layer
        Dense(4, activation='relu'),  # Second hidden layer
        Dense(output_dim, activation='softmax')  # Output layer for classification
    ])
    return model

# Plot training loss and accuracy
def plot_training_history(history):
    """
    Plot training and validation loss/accuracy curves over epochs.
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)  # Add gridlines

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Accuracy During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)  # Add gridlines

    plt.tight_layout()
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(true_labels, predictions, target_names):
    """
    Display a confusion matrix to visualize the classification results.
    """
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

# Predict a new example
def predict_new_example(model, scaler, target_names):
    """
    Predict the class of a new example.
    """
    print("\nPredicting a new example...")
    # Define a new sample (manually created or from the dataset)
    new_sample = np.array([[5.9, 3.0, 5.1, 1.8]])  # Example input features (normalized)

    # Normalize the sample using the same scaler
    new_sample_normalized = scaler.transform(new_sample)

    # Predict the class
    prediction = model.predict(new_sample_normalized)
    predicted_class = np.argmax(prediction)

    print(f"New Example Features: {new_sample[0]}")
    print(f"Predicted Class: {target_names[predicted_class]}")

# Train, evaluate, and predict using the ANN
def train_and_evaluate_model():
    """
    Load data, train the ANN model, evaluate performance, and visualize results.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, target_names, scaler = load_and_preprocess_data()

    # Build the model
    model = build_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Make predictions
    predictions = np.argmax(model.predict(X_test), axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Display predictions
    print("\nPredictions on test data:")
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        print(f"Sample {i+1}: True Label = {target_names[true]}, Predicted Label = {target_names[pred]}")

    # Plot training curves
    plot_training_history(history)

    # Plot confusion matrix
    print("Confusion Matrix:")
    plot_confusion_matrix(true_labels, predictions, target_names)

    # Predict a new example
    predict_new_example(model, scaler, target_names)

# Main function to execute the program
if __name__ == "__main__":
    try:
        print("Starting the ANN training and evaluation...")
        train_and_evaluate_model()
    except ImportError as e:
        print(f"Error: {e}. Please install TensorFlow and required libraries.")
