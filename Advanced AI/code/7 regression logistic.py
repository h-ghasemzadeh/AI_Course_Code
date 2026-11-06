import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Create the dataset
data = {
    'Heart Rate': [50, 50, 50, 50, 70, 70, 90, 90, 90, 90, 90],
    'Heart Attack': [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Prepare the data
X = df[['Heart Rate']]
y = df['Heart Attack']

# Train the logistic regression model
model = LogisticRegression()    
model.fit(X, y)

# Generate values for prediction
x_values = np.linspace(40, 120).reshape(-1, 1)
y_pred = model.predict_proba(x_values)[:, 1]

prediction_100 = model.predict_proba(np.array([[100]]))[0, 1]
print(f"Prediction of heart attack probability for heart rate 100: {prediction_100:.2f}") 

# Retrieve the coefficients for the logistic function formula
coef = model.coef_[0][0]        # Teta1 
intercept = model.intercept_[0] # Teta0

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(x_values, y_pred, color='red', label='Logistic Regression')
plt.xlabel('Heart Rate')
plt.ylabel('Probability of Heart Attack')
plt.title('Logistic Regression for Heart Attack Prediction')
plt.legend()

# Display formula on the plot
formula_text = f'P(y=1|x) = 1 / (1 + exp(-({coef:.4f} * x + {intercept:.4f})))'
plt.text(50, 0.2, formula_text, fontsize=12, color='purple', bbox=dict(facecolor='white', alpha=0.6))

plt.grid(True)
plt.show()

