import socketio
import eventlet
import eventlet.wsgi
import cv2
import numpy as np
import base64
from os import path
from flask import Flask

from agent import Agent
from preprocessor import process

sio = socketio.Server()
app = Flask(__name__)

dqn_agent = Agent()
dqn_agent.load('./model/autonomous.h5')
episode = 0
counter = 0
timestep = 0
cur_control = [0, 0]
buffer = [None, None, None] # last state, last action, last reward
INTERVAL = 15
INIT_THRESHOLD = 3
FINISH_THRESHOLD = 2.5
TRAIN = True


@sio.on('telemetry')
def telemetry(sid, data):
    global counter
    global timestep
    global episode
    global buffer
    global cur_control
    if data:
        timestep += 1
        if timestep % INTERVAL == 0:
            counter += 1
            print('episode {}, step {}'.format(episode, counter))
            cte = float(data['cte'])
            if counter <= INIT_THRESHOLD:
                cur_control = [0.0, 0.2]
                new_state = process(cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR))
                buffer[0], buffer[1], buffer[2] = new_state, 0, dqn_agent.get_reward(cte)
            else:
                state, action, reward = buffer[0], buffer[1], dqn_agent.get_reward(cte)
                new_state = process(cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR))
                if -FINISH_THRESHOLD < cte < FINISH_THRESHOLD:
                    new_action = 0
                    if TRAIN:
                        new_action = dqn_agent.act(new_state)
                        # buffer the current result
                        buffer[0], buffer[1], buffer[2] = new_state, new_action, dqn_agent.get_reward(cte)
                        dqn_agent.learn(state, action, reward, new_state)
                    else:
                        new_action = dqn_agent.act_greedy(new_state)
                    cur_control = Agent.ACTION_SPACE[new_action]
                else:
                    # new_state = np.zeros((60, 240, 1))
                    dqn_agent.learn(state, action, reward, new_state)
                    reset()
        send_control(cur_control[0], cur_control[1])
    else:
        sio.emit('manual', data={}, skip_sid=True)


def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': steering_angle,
            'throttle': throttle
        },
        skip_sid=True)


def reset():
    global episode
    global timestep
    global buffer
    global dqn_agent
    global cur_control
    global counter
    episode += 1
    print('reset, start episode: {}'.format(episode))
    timestep, counter = 0, 0
    buffer = [None, None, None]
    cur_control = [0.0, 0.0]
    if episode % 10 == 0:
        dqn_agent.save('./model/autonomous.h5')
    sio.emit('reset', {})


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
