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


class SiameseModel2(Model):
    def __init__(self, input_shape=Config.IMAGE_SHAPE, **kargs):
        super().__init__(**kargs)
        self.network = Sequential([
            layers.Input(shape=input_shape),
            EfficientNetV2B0(include_top=False, weights="imagenet"), # backbone
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(256)
        ])
    
    def call(self, anchor, positive, negative):
        anchor_pred = self.network(anchor)
        positive_pred = self.network(positive)
        negative_pred = self.network(negative)
        return anchor_pred, positive_pred, negative_pred