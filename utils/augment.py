import math
import random
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from . import common


class RandomZoominAndOut(layers.Layer):
    def __init__(self, factor: Tuple[float, float], **kwarg):
        super().__init__(**kwarg)
        self.minval = min(factor)
        self.maxval = max(factor)

    def call(self, x: tf.Tensor):
        assert len(x.shape) == 3 or len(x.shape) == 4, "tensor's dim have to 3 or 4"
        # assign height and width depends on tensor shape
        shape = x.shape[:2] if len(x.shape) == 3 else x.shape[1:3]
        zoom_ratio = tf.random.uniform(
            [], minval=self.minval, maxval=self.maxval, dtype=tf.float32
        )
        new_shape = tf.math.round(tf.cast(shape, tf.float32) * zoom_ratio)
        new_shape = tf.cast(new_shape, tf.int32)
        image = tf.image.resize(x, new_shape)
        image = tf.image.resize(image, shape)

        return image


def zoomout_and_transition(image, min_scale=0.5):
    # resize image
    height, width = image.shape[:2]
    scale = random.uniform(min_scale, 1.0)
    new_height = round(height * scale)
    new_width = round(width * scale)
    new_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # translate by padding
    dw = width - new_width
    dh = height - new_height
    left = round(dw * random.random())
    right = dw - left
    top = round(dh * random.random())
    bottom = dh - top
    new_image = cv2.copyMakeBorder(
        new_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return new_image, scale, (left, top)


def make_deck_image(images, padding_value=(114, 114, 114)):
    divisor = common.get_factors(len(images), 3)
    length = len(images)
    if len(divisor) == 0:
        length = int(math.ceil(len(images) / 10) * 10)
        divisor = common.get_factors(length, 3)

    row = random.choice(divisor)
    col = length // row

    height, width = images[0].shape[:2]
    sx = np.arange(0, col * width, width)
    sy = np.arange(0, row * height, height)
    sx, sy = np.meshgrid(sx, sy)
    offset = np.stack([sx, sy], axis=-1).reshape(-1, 2)
    if len(images) < length:
        diff = length - len(images)
        offset = offset[:-diff]

    total_image = []
    for row_index in range(row):
        start = row_index * col
        end = (row_index + 1) * col
        row_images = images[start:end]
        if len(row_images) < col:
            padding_image = np.full(images[0].shape, padding_value, dtype=np.uint8)
            num_padding = col - len(row_images)
            for _ in range(num_padding):
                row_images.append(padding_image.copy())
        total_image.append(np.concatenate(row_images, axis=1))

    return np.concatenate(total_image, axis=0), offset
