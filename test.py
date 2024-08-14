import os
import cv2
import random
import time
from typing import Tuple, Iterator

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from config import Config
import common
import custom

def main():
    image_path = common.extract_images(f"{Config.IMAGE_PATH}")
    id_list = list(map(common.get_filename, image_path))

    dataset = tf.data.Dataset.from_tensor_slices(image_path)
    dataset = dataset.map(common.load_image_to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(common.preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

    batch_size = 32
    model = custom.models.EmbeddingModel()
    matrix = model.make_matrix(dataset, batch_size=batch_size)

    hit = 0    
    for i, data in enumerate(dataset.batch(batch_size)):
        augment = common.augmentation(data)
        preds = model.pred(augment, batch_size=batch_size)

        for j, pred in enumerate(preds):
            result = tf.math.reduce_sum(tf.math.square(matrix - pred), axis=-1)
            min_index = np.argmin(result.numpy())
            if min_index == i * batch_size + j:
                hit += 1
    print(hit / len(dataset) * 100)


if __name__ == "__main__":
    main()