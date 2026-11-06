import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym  # Using Gymnasium (newer version of OpenAI Gym)

# Initialize the FrozenLake environment
# is_slippery=True makes the environment stochastic (ice is slippery)
# render_mode="rgb_array" enables rendering capability
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")

# Get environment dimensions
state_size = env.observation_space.n  # Number of possible states (16 for 4x4 grid)
action_size = env.action_space.n     # Number of possible actions (4: Left, Down, Right, Up)

# Initialize Q-table with zeros
# Q-table stores expected rewards for state-action pairs
Q = np.zeros((state_size, action_size))

# Hyperparameters for SARSA algorithm
alpha = 0.8         # Learning rate (how much we update Q-values)
gamma = 0.95        # Discount factor (importance of future rewards)
epsilon = 1.0       # Initial exploration rate (start with 100% random actions)
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Rate at which exploration decreases
episodes = 2000     # Number of training episodes
max_steps = 100     # Maximum steps per episode

# List to store rewards for each episode
rewards = []

# SARSA Algorithm Implementation
for episode in range(episodes):
    # Reset environment and get initial state
    state = env.reset()[0]  # [0] gets the initial state from the reset tuple
    done = False
    total_reward = 0

    # Select initial action using epsilon-greedy policy
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Random action (exploration)
    else:
        action = np.argmax(Q[state, :])    # Best known action (exploitation)

    for _ in range(max_steps):
        # Take the action and observe the outcome
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Episode ends if either condition is True

        # Select next action (on-policy)
        if np.random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()  # Random action
        else:
            next_action = np.argmax(Q[next_state, :])  # Best known action

        # SARSA Q-value update formula:
        # Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        # Transition to next state and action
        state = next_state
        action = next_action
        total_reward += reward

        if done:  # Episode terminated (reached goal or fell in hole)
            break

    # Decay epsilon to reduce exploration over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

# Calculate moving average of rewards (window size=100)
avg_rewards = np.convolve(rewards, np.ones(100)/100, mode='valid')

# Plot the learning progress
plt.figure(figsize=(10, 5))
plt.plot(avg_rewards)
plt.title("Average Reward Over Last 100 Episodes (SARSA on FrozenLake)")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()

epsilons = []
current_eps = 1.0
for _ in range(episodes):
    epsilons.append(current_eps)
    current_eps = max(0.01, current_eps * 0.995)

plt.plot(epsilons)
plt.title("Decay of ε (Exploration Rate)")
plt.xlabel("Episode")
plt.ylabel("ε Value")
plt.show()