# 🚀 Q-Learning for FrozenLake Environment

## 📌 Introduction

This project implements **Q-Learning**, a reinforcement learning algorithm, to solve the **FrozenLake-v1** environment using OpenAI Gym. The agent learns to navigate the slippery frozen lake and reach the goal while avoiding holes.

## 🎯 Key Features

- **Q-Learning Algorithm** for reinforcement learning
- **ε-Greedy Exploration** strategy to balance exploration and exploitation
- **Q-Table Update Mechanism** for optimal action selection
- **Customizable Parameters** for fine-tuning learning rate, discount factor, and exploration decay
- **Agent Performance Evaluation** to measure training success

## 🛠️ Installation & Requirements

Ensure you have the required dependencies installed:

```sh
pip install numpy gym
```

If you encounter any issues with NumPy, upgrade it:

```sh
pip install --upgrade numpy
```

## 📝 Implementation Details

### 🔹 Q-Learning Algorithm

```python
import numpy as np
import gym

# Patch NumPy for compatibility with Gym
if not hasattr(np, 'bool8'):
    np.bool8 = bool  # Force NumPy to use standard bool

# Initialize FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Initialize Q-Table
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Q-Learning Parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 5000
max_steps = 100

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        # Choose action using ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-Table
        if not terminated:
            q_table[state, action] += learning_rate * (
                reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
            )
        else:
            q_table[state, action] += learning_rate * (reward - q_table[state, action])

        state = new_state
        if done:
            break

    # Reduce epsilon to encourage exploitation over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training completed!")
```

### 🔹 Evaluating the Agent

```python
num_tests = 100
successes = 0

for _ in range(num_tests):
    state, _ = env.reset()
    done = False

    for _ in range(max_steps):
        action = np.argmax(q_table[state, :])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            if reward == 1:
                successes += 1
            break
        state = new_state

print(f"Success Rate: {successes / num_tests * 100:.2f}%")
```

## 📊 Expected Results

After training, the agent should achieve a success rate of around **70-90%**, depending on the randomness of the environment and parameter tuning.

## 🔧 Customization & Optimization

- Adjust the **learning rate, discount factor, epsilon decay, and number of episodes** to experiment with different training strategies.
- If training takes too long, reduce `episodes` or increase `epsilon_decay`.
- Modify `is_slippery` in `env.make("FrozenLake-v1", is_slippery=True)` to `False` for a deterministic environment.

## 💡 Troubleshooting

- **`np.bool8` Error?** Use the provided **NumPy patch** in the script.
- **Gym Not Working?** Ensure Gym is updated:
  ```sh
  pip install --upgrade gym
  ```
- **Low Success Rate?** Increase `episodes` and `discount_factor` or decrease `epsilon_decay` to allow better convergence.

## 📜 License

This project is open-source and available under the **MIT License**.

---

Happy Learning! 🚀

