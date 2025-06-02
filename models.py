
import numpy as np

def simulate_q_learning(alpha, beta, n_trials=200, reward_probs=None, initial_Q=(0.6, 0.6)):
    QA, QB = initial_Q
    Q_values = []
    choices = []
    rewards = []

    for t in range(n_trials):
        pA = np.exp(beta * QA) / (np.exp(beta * QA) + np.exp(beta * QB))
        choice = np.random.choice(['A', 'B'], p=[pA, 1 - pA])
        reward = np.random.choice([1, 0], p=reward_probs[t][choice])

        if choice == 'A':
            delta = reward - QA
            QA += alpha * delta
        else:
            delta = reward - QB
            QB += alpha * delta

        Q_values.append((QA, QB))
        choices.append(choice)
        rewards.append(reward)

    return choices, rewards, Q_values


def simulate_associability_rl(alpha, beta, eta, n_trials=200, reward_probs=None, initial_Q=(0.6, 0.6)):
    QA, QB = initial_Q
    kappaA, kappaB = 1.0, 1.0
    Q_values = []
    choices = []
    rewards = []

    for t in range(n_trials):
        pA = np.exp(beta * QA) / (np.exp(beta * QA) + np.exp(beta * QB))
        choice = np.random.choice(['A', 'B'], p=[pA, 1 - pA])
        reward = np.random.choice([1, 0], p=reward_probs[t][choice])

        if choice == 'A':
            delta = reward - QA
            kappaA = (1 - eta) * kappaA + eta * abs(delta)
            QA += alpha * kappaA * delta
        else:
            delta = reward - QB
            kappaB = (1 - eta) * kappaB + eta * abs(delta)
            QB += alpha * kappaB * delta

        Q_values.append((QA, QB))
        choices.append(choice)
        rewards.append(reward)

    return choices, rewards, Q_values
