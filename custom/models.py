import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential

from config import Config


class SiameseModel(Model):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.backbone = EfficientNetV2B0(include_top=False, weights="imagenet")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(256, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.output_layer = layers.Dense(256)

    def call(self, inputs):
        y1 = self.backbone(inputs)
        y2 = self.flatten(y1)
        y3 = self.dense1(y2)
        y4 = self.bn1(y3)
        y5 = self.dense2(y4)
        y6 = self.bn2(y5)
        return self.output_layer(y6)


class EmbeddingModel(Model):
    def __init__(self, input_shape=Config.IMAGE_SHAPE, **kargs):
        super().__init__(**kargs)
        self.model = EfficientNetV2B0(
            include_top=True,
            weights="imagenet",
            classifier_activation=None,
            input_shape=input_shape,
        )

    def call(self, anchor, positive, negative):
        pred_anchor = self.model(anchor)
        pred_positive = self.model(positive)
        pred_negative = self.model(negative)
        return pred_anchor, pred_positive, pred_negative

    def make_matrix(self, dataset, batch_size=64):
        matrix = []
        for batch in dataset.batch(batch_size):
            pred = self.model.predict(batch, batch_size=batch_size, verbose=False)
            matrix.append(pred)
        matrix = tf.concat(matrix, axis=0)
        return matrix

    def pred(self, inputs, batch_size):
        return self.model.predict(inputs, batch_size=batch_size, verbose=False)
