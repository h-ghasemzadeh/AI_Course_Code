import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics

# Define the dataset
data = {
    'Age': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Heart Rate': [50, 50, 50, 50, 70, 70, 90, 90, 90, 90, 90],
    'Heart Attack': [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Prepare the data
X = df[['Age', 'Heart Rate']]
y = df['Heart Attack']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# Check the model's parameters
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

theta_0 = model.intercept_[0]
theta_1 = model.coef_[0][0]
theta_2 = model.coef_[0][1]

print("Intercept (θ0):", theta_0)
print("Coefficient for Age (θ1):", theta_1)
print("Coefficient for Heart Rate (θ2):", theta_2)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and display performance metrics
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

# Example prediction for a person with Age = 20 and Heart Rate = 80
example_data = np.array([[20, 80]])
probability_heart_attack = model.predict_proba(example_data)[0][1]
print("Probability of Heart Attack for Age=20 and Heart Rate=80:", probability_heart_attack)

logistic_roc_auc = metrics.roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])


plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logistic_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()