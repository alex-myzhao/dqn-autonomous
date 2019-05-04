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
counter = 0
steer_buffer = 0
data_buffer = None
INTERVAL = 25


def get_action(steer):
    if steer < -15:
        return 0
    elif steer > 15:
        return 2
    else:
        return 1


def get_steer(action):
    steer_list = [-25.0, 0.0, 25.0]
    return steer_list[action]


def get_reward(cte, speed):
    return -(cte ** 2) * 10 + speed


def dqn(dqn_agent, data, next_data):
    state = cv2.resize(cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR), (160, 80))
    steer = float(data['steering_angle'])
    reward = float(next_data['cte'])
    # reward = get_reward(float(next_data['cte']), float(next_data['speed']))
    next_state = cv2.resize(cv2.imdecode(np.frombuffer(base64.b64decode(next_data['image']), np.uint8), cv2.IMREAD_COLOR), (160, 80))
    # print(get_action(steer), reward)
    return get_steer(dqn_agent.learn(state, get_action(steer), reward, next_state))


@sio.on('telemetry')
def telemetry(sid, data):
    global counter
    global steer_buffer
    global data_buffer
    global dqn_agent
    if data:
        if data_buffer == None:
            data_buffer = data
            return
        # safety control
        cte = float(data['cte'])
        if counter % INTERVAL == 0:
            v_pre, v = float(data_buffer['speed']), float(data['speed'])
            if v - v_pre < 0.1 and v < 0.1:
                reset()
                return
            if -2.2 < cte < 2.2:
                # DQN
                steer_buffer = dqn(dqn_agent, data_buffer, data)
            else:
                if cte >= 2.2:
                    steer_buffer = -25.0
                    data['steering_angle'] = -25.0
                elif cte <= -2.2:
                    steer_buffer = 25.0
                    data['steering_angle'] = 25.0
                data['ctx'] = 10
            data_buffer = data
        send_control(steer_buffer, 0.1)
        counter += 1
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
    global data_buffer
    print('reset')
    # reset all global settings
    data_buffer = None
    # storage buffered data
    sio.emit('reset', {})


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
