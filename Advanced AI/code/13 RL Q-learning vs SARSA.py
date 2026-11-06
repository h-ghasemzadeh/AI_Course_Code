import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Environment
env = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    render_mode="rgb_array"
)

state_size = env.observation_space.n
action_size = env.action_space.n

# Hyperparameters
alpha = 0.8
gamma = 0.95

episodes = 3000
max_steps = 100

epsilon0 = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995


# =====================================
# SARSA
# =====================================
Q_sarsa = np.zeros((state_size, action_size))
rewards_sarsa = []

epsilon = epsilon0

for episode in range(episodes):

    state = env.reset()[0]

    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_sarsa[state])

    total_reward = 0

    for step in range(max_steps):

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if np.random.rand() < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q_sarsa[next_state])

        # SARSA update
        Q_sarsa[state, action] += alpha * (
            reward
            + gamma * Q_sarsa[next_state, next_action]
            - Q_sarsa[state, action]
        )

        state = next_state
        action = next_action

        total_reward += reward

        if done:
            break

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    rewards_sarsa.append(total_reward)


# =====================================
# Q-Learning
# =====================================
Q_q = np.zeros((state_size, action_size))
rewards_q = []

epsilon = epsilon0

for episode in range(episodes):

    state = env.reset()[0]
    total_reward = 0

    for step in range(max_steps):

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        Q_q[state, action] += alpha * (
            reward
            + gamma * np.max(Q_q[next_state])
            - Q_q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    rewards_q.append(total_reward)


# =====================================
# Moving Average
# =====================================
window = 100

avg_sarsa = np.convolve(
    rewards_sarsa,
    np.ones(window) / window,
    mode="valid"
)

avg_q = np.convolve(
    rewards_q,
    np.ones(window) / window,
    mode="valid"
)

# =====================================
# Plot Comparison
# =====================================
plt.figure(figsize=(12,6))

plt.plot(
    avg_sarsa,
    label="SARSA",
    linewidth=2
)

plt.plot(
    avg_q,
    label="Q-Learning",
    linewidth=2
)

plt.xlabel("Episode")
plt.ylabel("Average Reward (100 Episodes)")
plt.title("SARSA vs Q-Learning on FrozenLake")
plt.grid(True)
plt.legend()

plt.show()

# =====================================
# Final Success Rates
# =====================================
print("\nFinal Results")
print("---------------------")
print("SARSA Average Reward:",
      np.mean(rewards_sarsa[-500:]))

print("Q-Learning Average Reward:",
      np.mean(rewards_q[-500:]))

env.close()