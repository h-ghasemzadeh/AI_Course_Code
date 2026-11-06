import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

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

# Train the logistic regression model
model = LinearRegression() # Optimization method L-BFGS 
model.fit(X, y)

# Check the model's parameters
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

theta_0 = model.intercept_[0]
theta_1 = model.coef_[0][0]
theta_2 = model.coef_[0][1]

print("Intercept (θ0):", theta_0)
print("Coefficient for Age (θ1):", theta_1)
print("Coefficient for Heart Rate (θ2):", theta_2)

# Example prediction for a person with Age = 20 and Heart Rate = 80
example_data = np.array([[20, 80]])
probability_heart_attack = model.predict_proba(example_data)[0][1]
print("Probability of Heart Attack for Age=20 and Heart Rate=80:", probability_heart_attack)

