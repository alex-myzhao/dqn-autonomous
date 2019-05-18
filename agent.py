import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import cv2

class Agent:
    ACTION_SPACE = [(0, 0.1), (-20, 0.1), (20, 0.1)]

    def __init__(self):
        # -- Basic settings related to the environment --
        self.num_actions = 3
        # -- TD trainer options --
        self.learning_rate = 0.01         # learning rate
        self.momentum = 0.8               # momentum
        self.batch_size = 4               # batch size
        self.decay = 0.00001              # learning rate decay
        # -- other options --
        self.gamma = 0.9                  # future discount for reward
        self.epsilon = 0.20               # epsilon during training
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.10
        self.start_learn_threshold = 20   # minimum number of examples in replay memory before learning
        self.experience_size = 10       # size of replay memory
        self.learning_steps_burnin = 20   # number of random actions the agent takes before learning
        self.learning_steps_total = 10000 # number of training iterations
        # -- Buffered model --
        self._model = self._build_model()
        # -- Replay memory --
        self._memory = [[] for _ in range(self.experience_size)]
        # -- Agent counter
        self.step = 0
        # loss record
        self.loss_rec = []

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(60, 240, 1)))
        model.add(keras.layers.Conv2D(8, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(16, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(self.num_actions, activation='linear'))

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay))
        model.summary()
        return model

    def _forward(self, state):
        prediction = self._model.predict(state)
        return prediction

    def _backward(self, state, target, epochs=1):
        history = self._model.fit(state, target, epochs=epochs, verbose=0)
        # print(history.history)
        print(history.history['loss'][-1])
        self.loss_rec.append(history.history['loss'][-1])
        # if self.step % 1000 == 0:
        #     np.save('./_out/loss.npy', self.loss_rec)

    def _remember(self, state, action, reward, next_state, debug=False):
        index = self.step % self.experience_size
        if self.step > self.experience_size:
            self.save_memory('./_out/memory.npy')
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
        self.epsilon = np.max([self.epsilon * self.epsilon_decay, self.min_epsilon])
        if self.step < self.learning_steps_burnin or np.random.rand() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            act_values = self._forward(np.array([state]))
            print(act_values)
            return np.argmax(act_values)

    def act_greedy(self, state):
        """ Always act based on the output of NN
        """
        act_values = self._forward(np.array([state]))
        print('Greedy: {}'.format(act_values))
        return np.argmax(act_values)

    def get_reward(self, cte):
        if abs(cte) >= 2.5:
            return -100
        else:
            return 1

    def load(self, name):
        self._model.load_weights(name)

    def save(self, name):
        self._model.save_weights(name)

    def save_memory(self, name):
        width, height = 240, 60
        serialized = []
        for index, m in enumerate(self._memory):
            print(m[0].shape)
            s_state = np.reshape(m[0], [width * height])
            s_next_state = np.reshape(m[3], [width * height])
            serialized.append(np.concatenate([s_state, [m[1]], [m[2]], s_next_state]))
        serialized = np.array(serialized)
        print(serialized.shape)
        np.save(name, serialized)

    def read_memory(self, name):
        width, height = 240, 60
        l = width * height
        serialized = np.load(name)
        self._memory = []
        for index, s in enumerate(serialized):
            state = np.reshape(s[0:l], [height, width, 1])
            action = s[l].astype(int)
            reward = s[l+1]
            next_state = np.reshape(s[l+2:], [height, width, 1])
            self._memory.append([state, action, reward, next_state])
        self.experience_size = len(serialized)
        print('load memory successfully')

    def offline_learn(self, iterations):
        for i in range(iterations):
            minibatch = random.sample(self._memory, self.batch_size)
            states = np.array([dp[0] for dp in minibatch])
            actions = np.array([dp[1] for dp in minibatch])
            rewards = np.array([dp[2] for dp in minibatch])
            next_states = np.array([dp[3] for dp in minibatch])
            q_hats = self._forward(next_states)
            td_targets = rewards + self.gamma * np.amax(q_hats)
            predictions = self._forward(states)
            for i in range(len(predictions)):
                predictions[i][actions[i]] = td_targets[i]
            self._backward(states, predictions, 20)
            if i % 100 == 0:
                np.save('./_out/loss.npy', self.loss_rec)
