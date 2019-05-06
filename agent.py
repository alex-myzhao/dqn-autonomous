import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import cv2

class Agent:
    ACTION_SPACE = [(0, 0.1), (-25, 0.1), (25, 0.1)]

    def __init__(self):
        # -- Basic settings related to the environment --
        self.num_actions = 3
        # -- TD trainer options --
        self.learning_rate = 0.001        # learning rate
        self.momentum = 0.8               # momentum
        self.batch_size = 8               # batch size
        self.l2_decay = 0.01              # L2 normalization
        # -- other options --
        self.temporal_window = 3          # number of pervious states an agent remembers
        self.gamma = 0.7                  # future discount for reward
        self.epsilon = 0.2                # epsilon during training
        self.start_learn_threshold = 20  # minimum number of examples in replay memory before learning
        self.experience_size = 3000       # size of replay memory
        self.learning_steps_burnin = 200  # number of random actions the agent takes before learning
        self.learning_steps_total = 10000 # number of training iterations
        # -- Buffered model --
        self._model = self._build_model()
        # -- Replay memory --
        self._memory = [[] for _ in range(self.experience_size)]
        # -- Agent counter
        self.step = 0

    def _build_model(self):
        # opt = keras.optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(batch_input_shape=(self.batch_size, 60, 240, 1)))
        model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(24, input_dim=64))
        model.add(keras.layers.Dense(24))
        model.add(keras.layers.Dense(self.num_actions, activation='linear'))
        # model.add(keras.layers.Activation('softmax'))
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model

    def _forward(self, state):
        prediction = self._model.predict(state)
        return prediction

    def _backward(self, state, target):
        history = self._model.fit(state, target, epochs=2, verbose=0)
        print(history.history)

    def _remember(self, state, action, reward, next_state, debug=True):
        index = self.step % self.experience_size
        self._memory[index] = (state, action, reward, next_state)
        if debug:
            mean, stdv = [120.9934, 18.8303]
            state = (state * stdv) + mean
            state = state.astype(int)
            cv2.imwrite('./_debug/{0:0>5}-{1}-{2}.png'.format(self.step, action, reward), state)
            # cv2.imwrite('./_debug/' + str(self.step) + '_' + str(action) + '_' + str(reward) + '.png', state)
            # cv2.imwrite('./_debug/' + str(self.step) + '_next' + '.png', next_state)

    def learn(self, state, action, reward, next_state):
        self._remember(state, action, reward, next_state)
        if self.step >= self.start_learn_threshold:
            minibatch = random.sample(self._memory[0:self.step], self.batch_size)
            states = np.array([dp[0] for dp in minibatch])
            actions = np.array([dp[1] for dp in minibatch])
            rewards = np.array([dp[2] for dp in minibatch])
            next_states = np.array([dp[3] for dp in minibatch])
            q_hats = self._forward(next_states)
            td_targets = rewards + self.gamma * np.amax(q_hats)
            predictions = self._forward(states)
            for i in range(len(predictions)):
                predictions[i][actions[i]] = td_targets[i]
            self._backward(states, predictions)
        self.step += 1

    def act(self, state):
        if self.step < self.learning_steps_burnin or np.random.rand() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            act_values = self._forward(np.array([state]))
            print(act_values)
            return np.argmax(act_values)

    def get_reward(self, cte):
        if abs(cte) >= 2.5:
            return -100
        elif abs(cte) >= 2.0:
            return -10
        elif abs(cte) >= 1.5:
            return 0
        elif abs(cte) >= 1.0:
            return 1
        elif abs(cte) >= 0.5:
            return 5
        else:
            return 10

    def load(self, name):
        self._model.load_weights(name)

    def save(self, name):
        self._model.save_weights(name)
