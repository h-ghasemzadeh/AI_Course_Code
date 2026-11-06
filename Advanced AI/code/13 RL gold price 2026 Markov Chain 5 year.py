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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# =========================================
# 1. Download 5 YEARS of REAL gold data
# =========================================

# Gold Futures symbol in Yahoo Finance
symbol = "GC=F"

# Date range (last 5 years)
end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

print("Downloading 5 years of gold price data...\n")

gold_data = yf.download(
    symbol,
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    auto_adjust=True
)

# =========================================
# 2. Extract closing prices safely
# =========================================

try:
    close_data = gold_data['Close']
    gold_prices = close_data.dropna().squeeze().to_numpy()

except Exception:
    # MultiIndex fallback
    close_data = gold_data[('Close', symbol)]
    gold_prices = close_data.dropna().to_numpy()

# Dates
dates = gold_data.index

# =========================================
# 3. Validate downloaded data
# =========================================

print("Downloaded Data Shape:", gold_prices.shape)

if len(gold_prices) < 200:
    raise ValueError("Insufficient historical data downloaded.")

# Current price
current_price = float(gold_prices[-1])

print(f"\nCurrent Gold Price: ${current_price:.2f}")

# =========================================
# 4. Define market states
# =========================================

def get_state(price_change):

    if price_change < -1.5:
        return "Sharp Drop"

    elif -1.5 <= price_change < 0:
        return "Mild Drop"

    elif 0 <= price_change < 1.5:
        return "Mild Rise"

    else:
        return "Sharp Rise"

states = [
    "Sharp Drop",
    "Mild Drop",
    "Mild Rise",
    "Sharp Rise"
]

state_indices = {
    state: idx
    for idx, state in enumerate(states)
}

# =========================================
# 5. Calculate percentage changes
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
# 6. Build transition matrix
# =========================================

transition_matrix = np.zeros((4, 4))

for i in range(len(state_sequence) - 1):

    current_state = state_indices[state_sequence[i]]
    next_state = state_indices[state_sequence[i + 1]]

    transition_matrix[current_state, next_state] += 1

# Normalize rows safely
row_sums = transition_matrix.sum(axis=1, keepdims=True)

# Prevent division by zero
row_sums[row_sums == 0] = 1

transition_matrix = transition_matrix / row_sums

# =========================================
# 7. Show transition matrix
# =========================================

print("\nTransition Matrix:\n")

transition_df = pd.DataFrame(
    transition_matrix,
    index=states,
    columns=states
)

print(transition_df)

# =========================================
# 8. Monte Carlo + Markov prediction
# =========================================

def simulate_future_prices(
    last_price,
    transition_matrix,
    state_sequence,
    days=180
):

    predicted_prices = [last_price]

    current_state = state_sequence[-1]

    for _ in range(days):

        probs = transition_matrix[
            state_indices[current_state]
        ]

        # Safety fallback
        if np.sum(probs) == 0:
            probs = np.ones(4) / 4

        # Select next state
        next_state = np.random.choice(
            states,
            p=probs
        )

        # Generate daily movement
        if next_state == "Sharp Drop":

            change = np.random.normal(-1.8, 0.5)

        elif next_state == "Mild Drop":

            change = np.random.normal(-0.4, 0.3)

        elif next_state == "Mild Rise":

            change = np.random.normal(0.4, 0.3)

        else:

            change = np.random.normal(1.8, 0.5)

        # Limit extreme values
        change = np.clip(change, -3.0, 3.0)

        # Compute new price
        new_price = predicted_prices[-1] * (
            1 + change / 100
        )

        predicted_prices.append(new_price)

        current_state = next_state

    return predicted_prices

# =========================================
# 9. Run multiple simulations
# =========================================

simulation_count = 100
days_to_predict = 180

all_simulations = []

for _ in range(simulation_count):

    sim_prices = simulate_future_prices(
        current_price,
        transition_matrix,
        state_sequence,
        days=days_to_predict
    )

    all_simulations.append(sim_prices)

all_simulations = np.array(all_simulations)

# Average prediction
mean_prediction = np.mean(all_simulations, axis=0)

# Confidence bands
upper_band = np.percentile(all_simulations, 90, axis=0)
lower_band = np.percentile(all_simulations, 10, axis=0)

# =========================================
# 10. Future dates
# =========================================

future_dates = pd.date_range(
    start=dates[-1],
    periods=days_to_predict + 1,
    freq='D'
)

# =========================================
# 11. Plot results
# =========================================

plt.figure(figsize=(15, 8))

# Historical prices
plt.plot(
    dates[-365:],
    gold_prices[-365:],
    label='Historical Gold Price',
    linewidth=2
)

# Plot sample simulations
for i in range(10):

    plt.plot(
        future_dates,
        all_simulations[i],
        alpha=0.3
    )

# Mean prediction
plt.plot(
    future_dates,
    mean_prediction,
    '--',
    linewidth=3,
    label='Average Prediction'
)

# Confidence interval
plt.fill_between(
    future_dates,
    lower_band,
    upper_band,
    alpha=0.2,
    label='80% Confidence Range'
)

# Current price line
plt.axhline(
    y=current_price,
    linestyle=':',
    label=f'Current Price (${current_price:.2f})'
)

plt.title(
    'Gold Price Forecast Using 5-Year Markov Model',
    fontsize=16
)

plt.xlabel('Date')
plt.ylabel('Gold Price (USD/oz)')

plt.legend()
plt.grid(True)

plt.show()

# =========================================
# 12. Statistics
# =========================================

final_prices = all_simulations[:, -1]

mean_final = np.mean(final_prices)
max_final = np.max(final_prices)
min_final = np.min(final_prices)

expected_change = (
    (mean_final - current_price)
    / current_price
) * 100

print("\nPrediction Results")
print("-" * 50)

print(f"Current Gold Price: ${current_price:.2f}")

print(
    f"Average Predicted Price (6 Months): "
    f"${mean_final:.2f}"
)

print(
    f"Maximum Simulated Price: "
    f"${max_final:.2f}"
)

print(
    f"Minimum Simulated Price: "
    f"${min_final:.2f}"
)

print(
    f"Expected Change: "
    f"{expected_change:.2f}%"
)

print(
    f"90th Percentile Price: "
    f"${np.percentile(final_prices, 90):.2f}"
)

print(
    f"10th Percentile Price: "
    f"${np.percentile(final_prices, 10):.2f}"
)