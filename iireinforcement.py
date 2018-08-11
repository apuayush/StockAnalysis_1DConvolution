from keras.models import load_model

import numpy as np
import random
from collections import deque
import cnn
import sys


class Agent:
    def __init__(self, state_shape, is_eval=False, model_name=""):
        self.state_shape = state_shape  # for my model 100 days
        self.action_size = 3  # sit, buy, sell
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def _model(self):
        cnn_model = cnn.CNN(layers=[
            {
                'type': 'sepconv1D',
                'args': {
                    'filters': 32,
                    'kernel_size': 5,
                    'activation': 'relu',
                    'input_shape': self.state_size
                }
            },
            {
                'type': 'maxpool1D',
                'args': {
                    'pool_size': 2
                }
            },
            {
                'type': 'conv1D',
                'args': {
                    'filters': 100,
                    'kernel_size': 3,
                    'activation': 'relu'
                }
            },
            {
                'type': 'maxpool1D',
                'args': {
                    'pool_size': 2
                }
            },
            {
                'type': 'dropout',
                'args': {
                    'ratio': 0.15
                }
            },
            {
                'type': 'flatten',
                'args': None
            },
            {
                'type': 'dense',
                'args': {
                    'output': 250
                }
            },
            {
                'type': 'dropout',
                'args': {
                    'ratio': 0.2
                }
            },
            {
                'type': 'activation',
                'args': {
                    'function': 'relu'
                }
            },
            {
                'type': 'dense',
                'args': {
                    'output': self.action_size
                }
            },
            {
                'type': 'activation',
                'args': {
                    'function': 'linear'
                }
            },
        ])
        cnn_model.build_model()
        cnn_model.compile()
        return cnn_model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class Reinforcement_train:
    def __init__(self, X, episodes , mean, std):
        self.agent = Agent(X[0].shape)
        self.perform_action(X, episodes)
        self.mean = mean
        self.std = std

    def unscale(self, Y):
        return Y * self.std + self.mean

    def perform_action(self, X, episodes):
        for ep in range(episodes):
            print("chunk completed"+str(ep)+"/"+episodes)
            state = X[0]

            total_profit = 0.0

            self.agent.inventory = []

            for chunk in range(len(X)-1):
                action = self.agent.act(state)
                # for action = 0 sit
                next_state = X[chunk+1]
                reward = 0

                if action == 1:  # buy
                    self.agent.inventory.append(X[chunk][0])
                    print("Buy: " + self.unscale(X[chunk][0]))

                elif action == 2 and len(self.agent.inventory) > 0:  # sell
                    bought_price = self.agent.inventory.pop(0)
                    reward = max(X[chunk][0] - bought_price, 0)
                    total_profit += X[chunk][0] - bought_price
                    print("Sell: " + self.unscale(X[chunk][0]) + " | Profit: " + self.unscale(X[chunk][0] - bought_price))

                    self.agent.memory.append((state, action, reward, next_state))
                    state = next_state

            print("Total Profit: " + self.unscale(total_profit))

            if len(self.agent.memory) > 32:
                self.agent.expReplay(batch_size=32)

        if ep % 20 == 0:
            agent.model.save("models/model_ep" + str(e))





