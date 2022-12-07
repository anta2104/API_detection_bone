
from flask import Flask, render_template, request
from flask import Markup
from flask_cors import CORS, cross_origin
import os
import base64

import random

from PIL import Image
import tensorflow as tf

import cv2
import torch
from numpy import random
import numpy as np
from fracture import process_output


# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
app.config['UPLOAD_FOLDER'] = "static"
cors = CORS(app)

# path to modelf
model = tf.saved_model.load("my_model_centernet_1/saved_model")

detect_fn = model.signatures['serving_default']

import json 
def load_labels_map(label_url):
    # read JSON file
    a =  open(label_url)
    data = json.load(a)
    if data:
        return data['labels']
    return []
label_maps = load_labels_map('label_map_bone.pbtxt')


@app.route("/", methods=['POST'])
def home_page():
    image = request.files['file']
    if image:
        # Lưu file
        print(image.filename)
        print(app.config['UPLOAD_FOLDER'])
        source = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        print("Save = ", source)
        image.save(source)

        img = Image.open(image)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        results = detect_fn(input_tensor)

        targetSize = { 'w': 0, 'h': 0 }
        targetSize['h'] = img.shape[0]
        targetSize['w'] = img.shape[1]

        output = process_output(results, 0.5, targetSize, label_maps)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = np.array(output['fracture'])
        font = cv2.FONT_HERSHEY_PLAIN
        height, width, channels = img.shape
        for i in range(len(boxes)):
                    x, y, w, h, j= boxes[i]
                    # x = x /width
                    # y = y / height
                    # w = w / width
                    # h = h / height
                    label = 'Fracture'
                    color = (0,0,255)
                    score = str(round(j,4))
                    cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), color, 2)
                    cv2.putText(img, label, (int(x), int(y - 10)), font, 1, color, 1)
                    cv2.putText(img, score, (int(x + 75), int(y - 10)), font, 1, color, 1)
                    
    cv2.imwrite('test.jpg' , img)
    img_string = base64.b64encode(img)  
    data = {
        'image_base64': str(img_string)
    }

    response = app.response_class(response=json.dumps(data),
                status=200,
                mimetype='application/json')

    return response



if __name__ == '__main__':
    app.run( port='3001', debug=True)