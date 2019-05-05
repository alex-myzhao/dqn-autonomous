import socketio
import eventlet
import eventlet.wsgi
import cv2
import numpy as np
import base64
from os import path
from flask import Flask

from agent import Agent

sio = socketio.Server()
app = Flask(__name__)

dqn_agent = Agent()
episode = 0
counter = 0
timestep = 0
cur_control = [0, 0]
buffer = [None, None] # last state, last action
INTERVAL = 25
INIT_THRESHOLD = 3
FINISH_THRESHOLD = 2.5


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
            if counter <= INIT_THRESHOLD:
                cur_control = [0.0, 0.2]
                new_state = cv2.resize(cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR), (160, 80))
                buffer[0], buffer[1] = new_state, 0
            else:
                cte = float(data['cte'])
                if -FINISH_THRESHOLD < cte < FINISH_THRESHOLD:
                    state, action = buffer[0], buffer[1]
                    new_state = cv2.resize(cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR), (160, 80))
                    reward = dqn_agent.get_reward(cte)
                    new_action = dqn_agent.act(new_state)
                    # buffer the current result
                    buffer[0], buffer[1] = new_state, new_action
                    dqn_agent.learn(state, action, reward, new_state)
                    cur_control = Agent.ACTION_SPACE[new_action]
                else:
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
    buffer = [None, None]
    cur_control = [0.0, 0.0]
    if episode % 10 == 0:
        dqn_agent.save('./model/autonomous.h5')
    sio.emit('reset', {})


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
