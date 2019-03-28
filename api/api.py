import trainer.model as m
from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import tensorflow as tf
import math

model = m.Model().get_model()

api = Flask(__name__)

@api.route('/', methods=['POST'])
def inference():
    image = np.fromstring(request.get_data(), np.uint8)
    image = cv.imdecode(image, cv.IMREAD_GRAYSCALE)
    image = cv.copyMakeBorder(image, int((400-np.shape(image)[0])/2), math.ceil((400-np.shape(image)[0])/2), int((400-np.shape(image)[1])/2), math.ceil((400-np.shape(image)[1])/2), cv.BORDER_CONSTANT, 0)
    image = np.reshape(image, (400, 400, 1))
    image = np.concatenate((image, image, image), axis=2)
    image = np.reshape(image, (1, 400, 400, 3))

    generator = model.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(image, batch_size=1, shuffle=False)
    )
    result = next(generator)

    response = {
        'probability': result['dense'][0],
        'classification': round(result['dense'][0])
    }
    return str(response), 200

api.run()