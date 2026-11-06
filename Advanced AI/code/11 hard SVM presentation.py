from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate a synthetic dataset
X, y = make_blobs(n_samples=50, centers=2, random_state=42)
y = 2 * y - 1  # Convert labels to {-1, 1}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the hard-margin SVM
hard_margin_svm = SVC(C=1e6, kernel='linear')
hard_margin_svm.fit(X_train, y_train)

# Evaluate the model
y_pred = hard_margin_svm.predict(X_test)
print(classification_report(y_test, y_pred))
