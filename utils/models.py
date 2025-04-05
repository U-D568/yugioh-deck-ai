import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0

import torch
import torch.nn as nn
from ultralytics.nn.modules import conv, block
from ultralytics import YOLO
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import scale_boxes

from utils import common, head, ops, struct
from config import Config


class EmbeddingModel():
    def __init__(self, input_shape=Config.IMAGE_SHAPE, **kargs):
        self._backbone = EfficientNetV2B0(
            include_top=True,
            weights="imagenet",
            classifier_activation=None,
            input_shape=input_shape,
        )
        self._head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
        ])
        self.model = tf.keras.Sequential([self._backbone, self._head])

    def __call__(self, inputs):
        out = self.model(inputs)
        out = tf.math.l2_normalize(out, axis=-1)
        return out

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

    def make_matrix(self, dataset, batch_size=64):
        matrix = []
        for batch in dataset.batch(batch_size):
            pred = self.model.predict(batch, batch_size=batch_size, verbose=False)
            matrix.append(pred)
        matrix = tf.concat(matrix, axis=0)
        return matrix

    def predict(self, inputs, batch_size=None):
        if batch_size is None:
            batch_size = inputs.shape[0]
        return self.model.predict(inputs, batch_size=batch_size, verbose=False)


class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.preprocessor = common.EmbeddingPreprocessor()

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
            cropped[i] = self.preprocessor(cropped[i])
        cropped = tf.stack(cropped)
        return cropped


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_size = 256
        self.nc = 1

        # backbone
        self.layer0 = conv.Conv(3, 16, 3, 2)
        self.layer1 = conv.Conv(16, 32, 3, 2)
        self.layer2 = block.C2f(32, 32, 1, True)
        self.layer3 = conv.Conv(32, 64, 3, 2)
        self.layer4 = block.C2f(64, 64, 2, True)
        self.layer5 = conv.Conv(64, 128, 3, 2)
        self.layer6 = block.C2f(128, 128, 2, True)
        self.layer7 = conv.Conv(128, 256, 3, 2)
        self.layer8 = block.C2f(256, 256, 1, True)
        self.layer9 = block.SPPF(256, 256, 5)

        # head
        self.upsample = nn.Upsample(None, 2, "nearest")
        self.layer12 = block.C2f(384, 128, 1)
        self.layer15 = block.C2f(192, 64, 1)
        self.layer16 = conv.Conv(64, 64, 3, 2)
        self.layer18 = block.C2f(192, 128, 1)
        self.layer19 = conv.Conv(128, 128, 3, 2)
        self.layer21 = block.C2f(384, 256, 1)
        self.layer22 = head.Detect(
            nc=self.nc, ch=[64, 128, 256], embedding_size=self.embedding_size
        )
        self.layer22.stride = (8, 16, 32)

    def forward(self, x):
        # backbone
        y0 = self.layer0(x)
        y1 = self.layer1(y0)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        y8 = self.layer8(y7)
        y9 = self.layer9(y8)

        # head
        y10 = self.upsample(y9)
        y10 = torch.concat([y10, y6], dim=1)
        y10 = self.layer12(y10)

        y11 = self.upsample(y10)
        y11 = torch.concat([y11, y4], dim=1)
        y11 = self.layer15(y11)  # out

        y12 = self.layer16(y11)
        y13 = torch.concat([y12, y10], dim=1)
        y13 = self.layer18(y13)  # out

        y14 = self.layer19(y13)
        y15 = torch.concat([y14, y9], dim=1)
        y15 = self.layer21(y15)  # out

        result = self.layer22([y11, y13, y15])
        return result

    def load(self, path):
        state = torch.load(path)
        self.load_state_dict(state)

    def decode_bboxes(self, x):
        shape = x[0].shpae
        x_cat = torch.cat([xi.view(shape[0], shape[1], -1) for xi in x], 2)
        anchors, strides = make_anchors(x_cat, self.strides, 0.5)
        anchors = anchors.transpose(0, 1)
        strides = strides.transpose(0, 1)

        box, embedding = x_cat.split(
            (shape[1] - self.embedding_size, self.embedding_size), 1
        )
        bbox = (
            self.dist2bbox(self.layer22.dfl(box), anchors.unsqueeze(0), True, 1)
            * strides
        )

        return torch.cat((bbox, embedding), 1)

    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py
    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def postprocess(
        self, pred_bbox, pred_embeds, input_shape, original_images
    ) -> struct.DetResult:
        pred_bbox, nms_mask = ops.non_max_suppression(pred_bbox)

        embeds = []
        for i, mask in enumerate(nms_mask):
            embeds.append(pred_embeds[i][:, mask])

        results = []
        for pred, embed, ori_img in zip(pred_bbox, embeds, original_images):
            pred[:, :4] = scale_boxes(input_shape, pred[:, :4], ori_img.shape)
            results.append(
                struct.DetResult(
                    ori_img, pred, embed.transpose(-1, -2).detach().cpu().numpy()
                )
            )
        return results
