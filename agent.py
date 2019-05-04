import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import cv2

class Agent:
    def __init__(self, options=None):
        # -- Basic settings related to the environment --
        self.num_actions = 3
        self.action_space = (0, 1, 2)
        # -- TD trainer options --
        self.learning_rate = 0.01                 # learning rate
        self.batch_size = 8              # batch size
        self.l2_decay = 0.01              # L2 normalization
        # -- other options --
        self.temporal_window = 3          # number of pervious states an agent remembers
        self.gamma = 0.7                  # future discount for reward
        self.epsilon = 0.3                # epsilon during training
        self.start_learn_threshold = 200  # minimum number of examples in replay memory before learning
        self.experience_size = 3000       # size of replay memory
        self.learning_steps_burnin = 1000 # number of random actions the agent takes before learning
        self.learning_steps_total = 10000 # number of training iterations
        # -- Buffered model --
        self._model = self._build_model()
        # -- Replay memory --
        self._memory = [[] for _ in range(self.experience_size)]
        # -- Agent counter
        self.step = 0

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(16, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.num_actions))
        model.add(keras.layers.Activation('softmax'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def feedforward(self, state):
        state = np.expand_dims(state, axis=0)
        prediction = self._model.predict(state)
        return prediction[0]

    def backward(self, state, target):
        state = np.expand_dims(state, axis=0)
        target = np.expand_dims(target, axis=0)
        self._model.fit(state, target, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state):
        index = self.step % self.experience_size
        cv2.imwrite('./_debug/' + str(self.step) + '_' + str(action) + '_' + str(reward) + '.png', state)
        # cv2.imwrite('./_debug/' + str(self.step) + 'next.png', next_state)
        self._memory[index] = (state, action, reward, next_state)

    def learn(self, state, action, reward, next_state):
        self.remember(state, action, reward, next_state)
        chosen_action = 1
        if self.step < self.start_learn_threshold:
            chosen_action = random.sample(self.action_space, 1)[0]
        else:
            # sample from the replay memory
            minibatch = random.sample(self._memory[0:self.step], self.batch_size)
            for state, action, reward, next_state in minibatch:
                q_hat = self.feedforward(next_state)
                td_target = reward + self.gamma * np.amax(q_hat)
                prediction = self.feedforward(state)
                prediction[action] = td_target
                self.backward(state, prediction)
            chosen_action = self.act(next_state)
        self.step += 1
        return chosen_action

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random exploration
            return random.sample(self.action_space, 1)[0]
        else:
            # Predict the reward value based on the given state
            act_values = self.feedforward(state)
            # Pick the action based on the predicted reward
            return np.argmax(act_values)

    def is_training(self):
        return self.step >= self.start_learn_threshold

    def save(self):
        pass

    def load(self):
        pass
