from flask import Flask, jsonify, send_file, request, Response
from flask_cors import CORS
from pprint import pprint
import json

import os
app = Flask(__name__)
CORS(app)

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.abspath(os.path.join(script_path, '..', '..', 'data'))
video_path = os.path.join(data_path, 'video')
label_path = os.path.join(data_path, 'label')


@app.route('/list_filenames')
def list_filenames():
    onlyfiles = [
        f for f in os.listdir(label_path)
        if os.path.isfile(os.path.join(label_path, f))
    ]
    prefixes = sorted([
        f.split('.')[0] for f in onlyfiles if f.split('.')[-1] == 'json'
    ])
    return jsonify(prefixes)


# Avoids caching from server-side. Alternatively, the user can disable cache in Chrome. But we never want to trust the user.
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route('/get_file/<dirname>/<filename>')
def get_file(dirname, filename):
    return send_file(os.path.join(data_path, dirname, filename))


@app.route('/save_file/<dirname>/<filename>', methods=['POST'])
def save_file(dirname, filename):
    save_path = os.path.join(data_path, dirname, filename)
    data = request.json
    with open(save_path, 'w') as f:
        json.dump(data, f)
    return Response('OK')


if __name__ == "__main__":
    app.run(port=12345)
