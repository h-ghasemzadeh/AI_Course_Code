import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Create FrozenLake environment
env = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    render_mode="rgb_array"
)

# Environment dimensions
state_size = env.observation_space.n
action_size = env.action_space.n

# Initialize Q-table
Q = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.8
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 2000
max_steps = 100

rewards = []

# ==========================
# Q-Learning Training
# ==========================
for episode in range(episodes):

    state = env.reset()[0]
    total_reward = 0

    for step in range(max_steps):

        # Epsilon-Greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # --------------------------
        # Q-Learning Update
        # --------------------------
        best_next_q = np.max(Q[next_state, :])

        Q[state, action] += alpha * (
            reward +
            gamma * best_next_q -
            Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay exploration rate
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    rewards.append(total_reward)

# ==========================
# Learning Curve
# ==========================
avg_rewards = np.convolve(
    rewards,
    np.ones(100)/100,
    mode='valid'
)

plt.figure(figsize=(10, 5))
plt.plot(avg_rewards)
plt.title("Q-Learning on FrozenLake")
plt.xlabel("Episode")
plt.ylabel("Average Reward (100 Episodes)")
plt.grid(True)
plt.show()

# ==========================
# Epsilon Decay Plot
# ==========================
eps_values = []

eps = 1.0

for _ in range(episodes):
    eps_values.append(eps)
    eps = max(
        epsilon_min,
        eps * epsilon_decay
    )

plt.figure(figsize=(10, 5))
plt.plot(eps_values)
plt.title("Exploration Rate Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True)
plt.show()

# ==========================
# Print learned Q-table
# ==========================
print("Learned Q-Table:")
print(Q)

env.close()