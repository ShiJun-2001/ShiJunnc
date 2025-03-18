import numpy as np

# Patch NumPy for compatibility with Gym
if not hasattr(np, 'bool8'):
    np.bool8 = bool  # Force NumPy to use standard bool

import gym  # Import Gym after patching NumPy

# 创建 FrozenLake 环境
env = gym.make("FrozenLake-v1", is_slippery=True)  # is_slippery=True 表示环境具有随机性

# 初始化 Q-Table
state_size = env.observation_space.n  # 状态数量
action_size = env.action_space.n  # 动作数量
q_table = np.zeros((state_size, action_size))  # Q-Table 初始化为 0

# Q-Learning 参数
learning_rate = 0.1   # 学习率 (alpha)
discount_factor = 0.99  # 折扣因子 (gamma)
epsilon = 1.0  # 探索率 (epsilon)
epsilon_decay = 0.995  # 探索率衰减
min_epsilon = 0.01  # 最小探索率
episodes = 5000  # 训练回合数
max_steps = 100  # 每个回合的最大步数

# 训练智能体
for episode in range(episodes):
    state, _ = env.reset()  # Gym API 更新，正确解包 state
    done = False

    for step in range(max_steps):
        # 选择动作 (ε-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            action = np.argmax(q_table[state, :])  # 选择最优动作

        # 执行动作，获取奖励和新状态
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 处理终止状态

        # 更新 Q-Table
        if not terminated:  # 如果未到终止状态，使用 Q-learning 公式
            q_table[state, action] += learning_rate * (
                reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
            )
        else:  # 终止状态的 Q 值只更新奖励
            q_table[state, action] += learning_rate * (reward - q_table[state, action])

        state = new_state  # 更新状态

        if done:
            break  # 终止当前回合

    # 逐步降低 epsilon（减少探索，增加利用）
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("训练完成！")

# 评估智能体
num_tests = 100  # 测试 100 次
successes = 0

for _ in range(num_tests):
    state, _ = env.reset()  # Gym API 更新
    done = False

    for _ in range(max_steps):
        action = np.argmax(q_table[state, :])  # 选择最优动作
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 处理终止状态

        if done:
            if reward == 1:  # 成功到达目标
                successes += 1
            break

        state = new_state  # 更新状态

print(f"测试成功率: {successes / num_tests * 100:.2f}%")