import gym
import numpy as np
import random

env = gym.make("FrozenLake-v0")
observation_space = env.observation_space.n
action_space = env.action_space.n
all_rewards = []

EPISODES = 10000
GAMMA = 0.99
INDEX = 1000
LEARNING_RATE = 0.1
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.1

class Agent:
    def __init__(self):
        self.exploration_rate = EXPLORATION_MAX
        self.q_table = np.zeros((observation_space, action_space))
    
    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return env.action_space.sample()
        
        return np.argmax(self.q_table[observation, :])
    
    def learn(self, state, action, reward, state_):
        current_reward = self.q_table[state, action]
        predicted_value_of_future  = max(self.q_table[state_, :])

        self.q_table[state, action] = current_reward + LEARNING_RATE * (reward + GAMMA * predicted_value_of_future  - current_reward)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


agent = Agent()

for i in range(EPISODES):
    state = env.reset()
    done = False
    score = 0
    step = 0
    rendering = False

    if i >= EPISODES - 5: rendering = True
    if rendering: print("--")

    while not done:
        if rendering: env.render()

        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.learn(state, action, reward, state_)
        state = state_

        score += reward
        step += 1
    
    if rendering: env.render()
    all_rewards.append(score)

value = 0
for i in range(len(all_rewards)):
    value += all_rewards[i]

    if i % INDEX == 0:
        print("average reward {}".format(value/INDEX))
        value = 0
