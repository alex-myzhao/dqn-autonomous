import socketio
import eventlet
import eventlet.wsgi
import cv2
import numpy as np
import base64
from os import path
from flask import Flask

sio = socketio.Server()
app = Flask(__name__)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        cte = float(data["cte"])  # cross track error
        imagedata = data["image"]
        # img = cv2.imdecode(np.frombuffer(base64.b64decode(imagedata), np.uint8), cv2.IMREAD_COLOR)
        # cv2.imwrite(path.join('_out', str(counter // 20) + '.png') , img)
        print(steering_angle, throttle, speed, cte)
        send_control(0, 0.2)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle,
            'throttle': throttle
        },
        skip_sid=True)


def reset():
    print('reset')
    # reset all global settings
    # storage buffered data
    sio.emit('reset', {})


if __name__ == "__main__":
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
