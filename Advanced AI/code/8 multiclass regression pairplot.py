import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Map target numbers to iris species names
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
species_names = [species_mapping[label] for label in y]

# Convert to DataFrame for easier plotting with species names
df = pd.DataFrame(X, columns=data.feature_names)
df['species'] = species_names  # Use species names instead of numbers

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train logistic regression model for multiclass classification
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
logreg.fit(X_train, y_train)

# Predict and evaluate
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=species_mapping.values()))

# Pairplot for 4D visualization using seaborn with species names
# plt.figure(figsize=(8, 5))
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle("Pairwise 2D Scatter Plots with Iris Species", y=0.99)
plt.show()
