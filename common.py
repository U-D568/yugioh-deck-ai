import math
import os
import random

import tensorflow as tf
from tensorflow.keras import layers, models

import custom
from config import Config


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def extract_images(path):
    images = []
    for curdir, subdir, files in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if not ext in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(curdir, file)
            images.append(path)
    return images


def get_filename(path):
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)
    return name


def load_image_to_tensor(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def make_matrix(model, dataset, batch_size=Config.BATCH_SIZE):
    matrix = []
    for data in dataset.batch(batch_size):
        pred = model(data)
        matrix.append(pred)
    matrix = tf.concat(matrix, axis=0)
    return matrix

def random_negative_selector(indices, image_pathes):
    # description: select negative data randomly
    # args:
    #    indices: indices of anchors from image_pathes
    #    image_pathes: total image pathes of dataset
    negative = []

    for index in indices:
        negative_index = -1
        while True:
            negative_index = random.randrange(len(image_pathes))
            if negative_index != index:
                break
        negative_path = image_pathes[negative_index]
        image_tensor = load_image_to_tensor(negative_path)
        image_tensor = preprocessing(image_tensor)
        image_tensor = augmentation(image_tensor)
        negative.append(image_tensor)

    return tf.stack(negative)


def hard_negative_selector(matrix, anchor, indices, image_pathes):
    # description: select negative value which has lowest distance in train dataset
    # args:
    #    matrix: prediction of all train dataset
    #    anchor: prediction of anchor
    #    indices: indices of anchor from image_pathes
    #    image_pathes: image_pathest of total dataset
    negative = []
    for encoding, anchor_index in zip(anchor, indices):
        min_index = -1
        min_value = math.inf

        if anchor_index > 0:
            lower_matrix = matrix[:anchor_index]
            lower_matrix = custom.losses.square_norm(lower_matrix, encoding)
            negative_index = tf.argmin(lower_matrix, axis=0, output_type=tf.int32)
            if lower_matrix[negative_index] < min_value:
                min_value = lower_matrix[negative_index]
                min_index = negative_index

        if anchor_index < len(image_pathes) - 1:
            upper_matrix = matrix[anchor_index + 1 :]
            upper_matrix = custom.losses.square_norm(upper_matrix, encoding)
            negative_index = tf.argmin(upper_matrix, axis=0, output_type=tf.int32)
            if upper_matrix[negative_index] < min_value:
                min_value = upper_matrix[negative_index]
                min_index = negative_index + anchor_index + 1

        negative_path = image_pathes[min_index]
        image = load_image_to_tensor(negative_path)
        image = preprocessing(image)
        image = augmentation(image)
        negative.append(image)
    
    return tf.stack(negative)


preprocessing = models.Sequential(
    [
        layers.Resizing(Config.IMAGE_SHAPE[0], Config.IMAGE_SHAPE[1]),
        layers.Rescaling(1.0 / 255),
    ]
)

augmentation = models.Sequential(
    [
        layers.Rescaling(255),
        layers.Resizing(Config.IMAGE_SHAPE[0], Config.IMAGE_SHAPE[1]),
        custom.augment.RandomZoominAndOut((0.4, 1)),
        # layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomContrast(0.4),
        layers.RandomBrightness(0.2),
        layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1)),
        layers.Rescaling(1.0 / 255),
    ]
)
