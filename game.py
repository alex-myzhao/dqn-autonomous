import numpy as np
import base64
import cv2
from preprocessor import process
from agent import Agent

class Environment:
    # Global constant flags
    INTERVAL = 15
    INIT_THRESHOLD = 3
    CRASH_THRESHOLD = 2.5

    def __init__(self, sio, mode='TRAIN'):
        self.episode = 0
        self.counter = 0
        self.timestep = 0
        self.current_control = [0.0, 0.2]
        # last state, last action, last reward
        self.buffer = [None, None, None]
        self.record = []
        self.mode = mode
        # dqn agent
        self.agent = Agent()
        self.agent.load('./model/offline_model.h5')
        # socket io connection
        self.sio = sio

    def play(self, data):
        cte = float(data['cte'])
        if self.tick_tack():
            new_state = process(cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR))
            if self.get_stage() == 'INIT':
                self.buffer = [new_state, 0, self.agent.get_reward(cte)]
            elif self.get_stage() == 'TEST' and not self.is_crash(cte):
                self.current_control = Agent.ACTION_SPACE[self.agent.act_greedy(new_state)]
            elif self.get_stage() == 'TEST' and self.is_crash(cte):
                self.reset()
            elif self.get_stage() == 'TRAIN' and not self.is_crash(cte):
                state, action, reward = self.buffer[0], self.buffer[1], self.agent.get_reward(cte)
                # new_action = self.agent.act(new_state)
                new_action = self.agent.act_with_guidence(new_state, cte)
                self.buffer = [new_state, new_action, self.agent.get_reward(cte)]
                self.agent.learn(state, action, reward, new_state)
                self.current_control = Agent.ACTION_SPACE[new_action]
            elif self.get_stage() == 'TRAIN' and self.is_crash(cte):
                state, action, reward = self.buffer[0], self.buffer[1], self.agent.get_reward(cte)
                self.agent.learn(state, action, reward, new_state)
                self.reset()
            else:
                print('Error: invalid stage')
        self.async_step(self.current_control)

    def async_step(self, action):
        self.sio.emit(
            'steer',
            data={
                'steering_angle': action[0],
                'throttle': action[1]
            },
            skip_sid=True)

    def reset(self):
        print('reset, start episode: {}'.format(self.episode))
        self.episode += 1
        self.record.append(self.counter)
        self.timestep, self.counter = 0, 0 # reset counters
        self.buffer = [None, None, None]
        self.current_control = [0.0, 0.2]
        if self.episode % 10 == 0:
            self.agent.save('./model/autonomous.h5')
            np.save('./_out/record.npy', self.record)
        self.sio.emit('reset', {})

    def tick_tack(self):
        self.timestep += 1
        if self.timestep % Environment.INTERVAL == 0:
            self.counter += 1
            print('episode {}, step {}'.format(self.episode, self.counter))
            return True
        else:
            return False

    def get_stage(self):
        if self.counter < Environment.INIT_THRESHOLD:
            return 'INIT'
        else:
            return self.mode

    def is_crash(self, cte):
        return abs(cte) > Environment.CRASH_THRESHOLD
