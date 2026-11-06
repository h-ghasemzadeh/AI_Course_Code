import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime

# =========================================
# 1. Download REAL gold price data
# =========================================

# Gold Futures symbol in Yahoo Finance
symbol = "GC=F"

# Download data
gold_data = yf.download(
    symbol,
    start="2024-01-01",
    end=datetime.today().strftime('%Y-%m-%d'),
    auto_adjust=True
)

# =========================================
# 2. Extract closing prices safely
# =========================================

try:
    # Standard format
    close_data = gold_data['Close']

    # Convert to 1D numpy array
    gold_prices = close_data.dropna().squeeze().to_numpy()

except Exception:
    # MultiIndex fallback
    close_data = gold_data[('Close', symbol)]
    gold_prices = close_data.dropna().to_numpy()

# Dates
dates = gold_data.index

# Check data
print("Gold Prices Shape:", gold_prices.shape)

if len(gold_prices) < 10:
    raise ValueError("Not enough data downloaded.")

# Current price
current_price = float(gold_prices[-1])

print(f"\nCurrent Gold Price: ${current_price:.2f}")

# =========================================
# 3. Define market states
# =========================================

def get_state(price_change):

    if price_change < -1.0:
        return "Sharp Drop"

    elif -1.0 <= price_change < 0:
        return "Mild Drop"

    elif 0 <= price_change < 1.0:
        return "Mild Rise"

    else:
        return "Sharp Rise"

states = [
    "Sharp Drop",
    "Mild Drop",
    "Mild Rise",
    "Sharp Rise"
]

state_indices = {s: i for i, s in enumerate(states)}

# =========================================
# 4. Calculate daily percentage changes
# =========================================

price_changes = (
    np.diff(gold_prices)
    / gold_prices[:-1]
) * 100

# Convert changes to state sequence
state_sequence = [
    get_state(change)
    for change in price_changes
]

# =========================================
# 5. Build transition matrix
# =========================================

transition_matrix = np.zeros((4, 4))

for i in range(len(state_sequence) - 1):

    current_state = state_indices[state_sequence[i]]
    next_state = state_indices[state_sequence[i + 1]]

    transition_matrix[current_state, next_state] += 1

# Normalize rows safely
row_sums = transition_matrix.sum(axis=1, keepdims=True)

# Avoid division by zero
row_sums[row_sums == 0] = 1

transition_matrix = transition_matrix / row_sums

print("\nTransition Matrix:\n")

print(pd.DataFrame(
    transition_matrix,
    index=states,
    columns=states
))

# =========================================
# 6. Predict future prices
# =========================================

def predict_future(
    last_price,
    transition_matrix,
    state_sequence,
    days=180
):

    predicted_prices = [last_price]

    # Start from last observed state
    current_state = state_sequence[-1]

    for _ in range(days):

        probs = transition_matrix[
            state_indices[current_state]
        ]

        # Safety check
        if np.sum(probs) == 0:
            probs = np.ones(4) / 4

        # Choose next state
        next_state = np.random.choice(
            states,
            p=probs
        )

        # Simulate movement
        if next_state == "Sharp Drop":
            change = np.random.uniform(-2.5, -1.0)

        elif next_state == "Mild Drop":
            change = np.random.uniform(-1.0, 0)

        elif next_state == "Mild Rise":
            change = np.random.uniform(0, 1.0)

        else:
            change = np.random.uniform(1.0, 2.5)

        # New price
        new_price = predicted_prices[-1] * (
            1 + change / 100
        )

        predicted_prices.append(new_price)

        current_state = next_state

    return predicted_prices

# Run prediction
predicted_prices = predict_future(
    current_price,
    transition_matrix,
    state_sequence,
    days=180
)

# =========================================
# 7. Create future dates
# =========================================

future_dates = pd.date_range(
    start=dates[-1],
    periods=181,
    freq='D'
)

# =========================================
# 8. Plot results
# =========================================

plt.figure(figsize=(14, 7))

# Historical prices
plt.plot(
    dates[-180:],
    gold_prices[-180:],
    label='Historical Gold Price'
)

# Predicted prices
plt.plot(
    future_dates,
    predicted_prices,
    '--',
    label='6-Month Prediction'
)

# Current price line
plt.axhline(
    y=current_price,
    linestyle=':',
    label=f'Current Price (${current_price:.2f})'
)

plt.title('Gold Price Prediction Using Markov Chain')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD/oz)')

plt.legend()
plt.grid(True)

plt.show()

# =========================================
# 9. Prediction statistics
# =========================================

print("\nPrediction Results")
print("-" * 40)

print(f"Starting Price: ${current_price:.2f}")

print(
    f"Predicted Price After 6 Months: "
    f"${predicted_prices[-1]:.2f}"
)

print(
    f"Maximum Predicted Price: "
    f"${max(predicted_prices):.2f}"
)

print(
    f"Minimum Predicted Price: "
    f"${min(predicted_prices):.2f}"
)

change_percent = (
    (predicted_prices[-1] - current_price)
    / current_price
) * 100

print(
    f"Expected Change: "
    f"{change_percent:.2f}%"
)