# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Accuracy of the Random Forest Classifier: {accuracy:.2f}")

# Visualize one of the decision trees from the Random Forest
plt.figure(figsize=(12, 8))
plot_tree(
    rf_clf.estimators_[0],  # Select the first decision tree in the forest
    filled=True,
    feature_names=data.feature_names,
    class_names=data.target_names,
)
plt.title(f"Visualization of One Decision Tree in the Random Forest\nAccuracy: {accuracy:.2f}")
plt.show()
