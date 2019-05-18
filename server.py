import socketio
import eventlet
import eventlet.wsgi
import cv2
import numpy as np
import base64
from os import path
from flask import Flask

import game
from preprocessor import process

sio = socketio.Server()
app = Flask(__name__)
genv = game.Environment(sio, 'TRAIN')


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        genv.play(data)
    else:
        sio.emit('manual', data={}, skip_sid=True)


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
