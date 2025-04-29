from flask import Flask, request
import cv2
import numpy as np
from predictionModel.predictionModel import PredictionModel

app = Flask(__name__)

predictor = PredictionModel()

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['file']
    file_bytes = file.stream.read()

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    print(predictor.predict(img))
    return 'success'