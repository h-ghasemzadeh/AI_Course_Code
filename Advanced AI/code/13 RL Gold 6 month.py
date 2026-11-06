import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# 1. Generate synthetic historical data ending at $3360 today
np.random.seed(42)
days_history = 365  # 1 year of historical data
dates = pd.date_range(end=datetime.today(), periods=days_history, freq='D')

# Generate random walk with $3360 as last price
random_walk = np.cumsum(np.random.normal(0, 5, days_history))
gold_prices = random_walk - random_walk[-1] + 3360  # Force last price to be $3360

# 2. Define states based on daily % changes
def get_state(price_change):
    if price_change < -0.5:
        return "Sharp Drop"
    elif -0.5 <= price_change < 0:
        return "Mild Drop"
    elif 0 <= price_change < 0.5:
        return "Mild Rise"
    else:
        return "Sharp Rise"

states = ["Sharp Drop", "Mild Drop", "Mild Rise", "Sharp Rise"]

# 3. Calculate transition matrix
price_changes = np.diff(gold_prices) / gold_prices[:-1] * 100
state_sequence = [get_state(change) for change in price_changes]

# Initialize and populate transition matrix
transition_matrix = np.zeros((4, 4))
state_indices = {s: i for i, s in enumerate(states)}

for i in range(len(state_sequence)-1):
    current = state_indices[state_sequence[i]]
    next_ = state_indices[state_sequence[i+1]]
    transition_matrix[current, next_] += 1

# Normalize rows to get probabilities
transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

# 4. Prediction function
def predict_future(last_price, transition_matrix, days=180):
    current_state = get_state(0)  # Start from no-change state
    predicted_prices = [last_price]
    
    for _ in range(days):
        # Get next state probabilities
        probs = transition_matrix[state_indices[current_state]]
        next_state = np.random.choice(states, p=probs)
        
        # Generate price change based on state
        if next_state == "Sharp Drop":
            change = np.random.uniform(-2.0, -0.5)
        elif next_state == "Mild Drop":
            change = np.random.uniform(-0.5, 0)
        elif next_state == "Mild Rise":
            change = np.random.uniform(0, 0.5)
        else:  # Sharp Rise
            change = np.random.uniform(0.5, 2.0)
        
        new_price = predicted_prices[-1] * (1 + change/100)
        predicted_prices.append(new_price)
        current_state = next_state
    
    return predicted_prices

# 5. Run prediction starting from $3360
predicted_prices = predict_future(3360, transition_matrix)

# 6. Plot results
future_dates = pd.date_range(start=dates[-1], periods=181, freq='D')

plt.figure(figsize=(12, 6))
plt.plot(dates[-90:], gold_prices[-90:], label='Last 90 Days History', color='blue') 
plt.plot(future_dates, predicted_prices, label='6-Month Prediction', color='red', linestyle='--')
plt.axhline(y=3360, color='green', linestyle=':', label='Today ($3360)')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD/oz)')
plt.title('Gold Price Prediction Starting from $3360')
plt.legend()
plt.grid(True)
plt.show()

# Print key metrics
print(f"Predicted price after 6 months: ${predicted_prices[-1]:.2f}")
print(f"Maximum predicted price: ${max(predicted_prices):.2f}")
print(f"Minimum predicted price: ${min(predicted_prices):.2f}")