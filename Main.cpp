import numpy as np

# Patch NumPy for compatibility with Gym
if not hasattr(np, 'bool8') :
    np.bool8 = bool  # Force NumPy to use standard bool

import gym  # Import Gym after patching NumPy

    # ���� FrozenLake ����
    env = gym.make("FrozenLake-v1", is_slippery = True)  # is_slippery = True ��ʾ�������������

    # ��ʼ�� Q - Table
    state_size = env.observation_space.n  # ״̬����
    action_size = env.action_space.n  # ��������
    q_table = np.zeros((state_size, action_size))  # Q - Table ��ʼ��Ϊ 0

    # Q - Learning ����
    learning_rate = 0.1   # ѧϰ��(alpha)
    discount_factor = 0.99  # �ۿ�����(gamma)
    epsilon = 1.0  # ̽����(epsilon)
    epsilon_decay = 0.995  # ̽����˥��
    min_epsilon = 0.01  # ��С̽����
    episodes = 5000  # ѵ���غ���
    max_steps = 100  # ÿ���غϵ������

    # ѵ��������
    for episode in range(episodes) :
        state, _ = env.reset()  # Gym API ���£���ȷ��� state
        done = False

        for step in range(max_steps) :
            # ѡ����(�� - greedy)
            if np.random.rand() < epsilon:
action = env.action_space.sample()  # ���̽��
            else:
action = np.argmax(q_table[state, :])  # ѡ�����Ŷ���

# ִ�ж�������ȡ��������״̬
new_state, reward, terminated, truncated, _ = env.step(action)
done = terminated or truncated  # ������ֹ״̬

# ���� Q - Table
if not terminated:  # ���δ����ֹ״̬��ʹ�� Q - learning ��ʽ
q_table[state, action] += learning_rate * (
    reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
    )
else:  # ��ֹ״̬�� Q ֵֻ���½���
q_table[state, action] += learning_rate * (reward - q_table[state, action])

state = new_state  # ����״̬

if done:
break  # ��ֹ��ǰ�غ�

# �𲽽��� epsilon������̽�����������ã�
epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("ѵ����ɣ�")

# ����������
num_tests = 100  # ���� 100 ��
successes = 0

for _ in range(num_tests) :
    state, _ = env.reset()  # Gym API ����
    done = False

    for _ in range(max_steps) :
        action = np.argmax(q_table[state, :])  # ѡ�����Ŷ���
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # ������ֹ״̬

        if done:
if reward == 1 : # �ɹ�����Ŀ��
successes += 1
break

state = new_state  # ����״̬

print(f"���Գɹ���: {successes / num_tests * 100:.2f}%")