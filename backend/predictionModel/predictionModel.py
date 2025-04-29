from ultralytics import YOLO

class PredictionModel:
    def __init__(self):
        self.model = YOLO("./predictionModel/best.pt")

    def predict(self, image):
        return self.model.predict(image)