# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the base estimator (weak learner)
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# Initialize and train the AdaBoost Classifier
adaboost_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
adaboost_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = adaboost_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Accuracy of the AdaBoost Classifier: {accuracy:.2f}")

# Visualize the first weak learner (decision tree)
plt.figure(figsize=(12, 8))
plot_tree(
    adaboost_clf.estimators_[0],  # Select the first weak learner
    filled=True,
    feature_names=data.feature_names,
    class_names=data.target_names,
)
plt.title(f"Visualization of the First Weak Learner in AdaBoost\nAccuracy: {accuracy:.2f}")
plt.show()
