import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam


class BitFlipEnvironment:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.state = np.random.randint(2, size=num_bits)
        self.target = np.random.randint(2, size=num_bits)

    def reset(self):
        self.state = np.random.randint(2, size=self.num_bits)
        self.target = np.random.randint(2, size=self.num_bits)
        return self.state.copy()

    def step(self, action):
        self.state[action] = 1 - self.state[action]
        done = np.array_equal(self.state, self.target)
        reward = 1.0 if done else 0.0
        return self.state.copy(), reward, done


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, predict=False):
        if not predict and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)


if __name__ == "__main__":
    # Parameters
    num_bits = 4
    state_size = num_bits
    action_size = num_bits

    # Create environment and agent
    env = BitFlipEnvironment(num_bits)
    agent = DQNAgent(state_size, action_size)

    # Training the DQN agent
    episodes = 1000
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        sumRevords = 0
        print('episode:', str(episode))
        for time in range(num_bits):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sumRevords = sumRevords+reward
            if done:
                print("Episode: {}, Flips: {}, Score {}".format(episode, time, sumRevords))
                break


        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
