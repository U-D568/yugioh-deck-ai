import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from ultralytics import YOLO

from utils import *
from utils import common
from config import Config


class EmbeddingModel(Model):
    def __init__(self, input_shape=Config.IMAGE_SHAPE, matrix_path="", **kargs):
        super().__init__(**kargs)
        self.model = EfficientNetV2B0(
            include_top=True,
            weights="imagenet",
            classifier_activation=None,
            input_shape=input_shape,
        )
        self.matrix = np.load(matrix_path)

    def call(self, anchor, positive, negative):
        pred_anchor = self.model(anchor)
        pred_positive = self.model(positive)
        pred_negative = self.model(negative)
        return pred_anchor, pred_positive, pred_negative

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def make_matrix(self, dataset, batch_size=64):
        matrix = []
        for batch in dataset.batch(batch_size):
            pred = self.model.predict(batch, batch_size=batch_size, verbose=False)
            matrix.append(pred)
        matrix = tf.concat(matrix, axis=0)
        return matrix

    def pred(self, inputs, batch_size):
        result = []
        preds = self.model.predict(inputs, batch_size=batch_size, verbose=False)
        for pred in preds:
            distance = losses.square_norm(self.matrix, pred)
            min_index = np.argmin(distance.numpy())
            result.append(min_index)
        return result


class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def pred(self, inputs: list):
        result_list = self.model.predict(inputs)
        cropped = []
        for i, result in enumerate(result_list):
            for box in result.boxes:
                if box.conf < 0.5:
                    continue
                x1, y1, x2, y2 = list(map(round, box.xyxy.tolist()[0]))
                cropped.append(inputs[i][y1:y2, x1:x2, :])

        for i in range(len(cropped)):
            cropped[i] = common.preprocessing(cropped[i])
        cropped = tf.stack(cropped)
        return cropped