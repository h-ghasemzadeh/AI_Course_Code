import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_iris()
X = data.data[:, 2:4]  # Select only petal length and petal width
y = data.target

# Map target numbers to iris species names
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
species_names = [species_mapping[label] for label in y]

# Convert to DataFrame for easier plotting and visualization
df = pd.DataFrame(X, columns=['petal length (cm)', 'petal width (cm)'])
df['species'] = species_names  # Use species names instead of numbers

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train logistic regression model for multiclass classification with sag solver
logreg = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=200)
logreg.fit(X_train, y_train)

# Predict and evaluate
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=species_mapping.values()))

# Convert test labels and predictions to species names for plotting
y_test_names = [species_mapping[label] for label in y_test]
y_pred_names = [species_mapping[label] for label in y_pred]

# Plotting classification results for petal length and petal width
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test_names, style=y_pred_names, palette='viridis', s=100, edgecolor="k")
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Logistic Regression Classification: Petal Length vs Petal Width')
plt.legend(title="True Species")
plt.show()
