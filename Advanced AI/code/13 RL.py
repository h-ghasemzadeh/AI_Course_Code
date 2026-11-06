import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. تعریف حالت‌ها و ماتریس انتقال
states = ["Sunny", "Claudy", "Rainy"]
transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # از آفتابی به [آفتابی, ابری, بارانی]
    [0.3, 0.4, 0.3],  # از ابری به ...
    [0.2, 0.3, 0.5]   # از بارانی به ...
])

# 2. تابع تولید دنباله مارکوف
def generate_markov_sequence(initial_state, steps=100):
    current_state = initial_state
    sequence = [current_state]
    state_indices = {s: i for i, s in enumerate(states)}
    
    for _ in range(steps):
        prob = transition_matrix[state_indices[current_state]]
        current_state = np.random.choice(states, p=prob)
        sequence.append(current_state)
    
    return sequence

# 3. تولید دنباله و ذخیره نتایج
np.random.seed(42)  # برای تکرارپذیری
sequence = generate_markov_sequence("Sunny", 50)

# 4. تبدیل حالت‌ها به اعداد برای رسم نمودار
state_to_num = {"Sunny": 0, "Claudy": 1, "Rainy": 2}
numeric_sequence = [state_to_num[s] for s in sequence]

# 5. رسم نمودار
plt.figure(figsize=(10, 5))
plt.plot(numeric_sequence, 'o-', markersize=8, linewidth=2)
plt.yticks([0, 1, 2], states)
plt.xlabel("Day")
plt.ylabel("Weather Condition")
plt.title("Markov Chain, Changing weather")
plt.grid(True)
plt.show()