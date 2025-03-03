from concurrent import futures
from collections.abc import Callable, Iterable
from collections import deque
import datetime
import logging
import math
import os
import pickle
import random
import threading
import urllib

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import torch

import utils
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


def count_parameters(layers: torch.nn.ModuleList):
    total = 0
    for param in layers.parameters():
        count = 1
        for size in param.size():
            count *= size
        total += count
    return total


def load_image_to_tensor(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


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
            lower_matrix = utils.losses.square_norm(lower_matrix, encoding)
            negative_index = tf.argmin(lower_matrix, axis=0, output_type=tf.int32)
            if lower_matrix[negative_index] < min_value:
                min_value = lower_matrix[negative_index]
                min_index = negative_index

        if anchor_index < len(image_pathes) - 1:
            upper_matrix = matrix[anchor_index + 1 :]
            upper_matrix = utils.losses.square_norm(upper_matrix, encoding)
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


def xyxy2xywh(ary: np.array):
    shape = ary.shape
    assert shape[-1] == 4, "Invalid input shape. The last dimension size have to be 4."

    flatten = ary.reshape(-1, 4)
    height = flatten[:, 2] - flatten[:, 0]
    width = flatten[:, 3] - flatten[:, 1]

    flatten[:, 0] += width / 2
    flatten[:, 1] += height / 2
    flatten[:, 2] = width
    flatten[:, 3] = height

    xywh = np.reshape(flatten, shape)
    return xywh


def xywh2xyxy(ary: np.array):
    shape = ary.shape
    assert shape[-1] == 4, "Invalid input shape. The last dimension size have to be 4."

    flatten = ary.reshape(-1, 4)
    center = flatten[:, :2]
    wh = flatten[:, 2:]
    flatten[:, :2] = center - wh
    flatten[:, 2:] = center + wh

    xyxy = np.reshape(flatten, shape)
    return xyxy


def make_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def is_prime(num):
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def get_factors(num, min_value=1):
    result = []
    upper_bound = int(math.sqrt(num)) + 1
    for i in range(min_value, upper_bound):
        if num % i == 0:
            result.append(i)
            result.append(num // i)
    return result


def detector_preprocessing(inputs):
    # preprocess
    assert len(inputs.shape) == 4 or len(inputs.shape) == 3

    deck_image = torch.from_numpy(inputs).float()
    if len(inputs.shape) == 3:
        deck_image = deck_image.permute([2, 0, 1])
    else:
        deck_image = deck_image.permute([0, 3, 1, 2])

    return deck_image / 255.0


class EmbeddingPreprocessor:
    def __init__(self):
        self.preprocessing = models.Sequential(
            [
                layers.Resizing(224, 224),
                layers.Rescaling(1.0 / 255),
            ]
        )

    def __call__(self, inputs):
        return self.preprocessing(inputs)


class EmbeddingAugmentation:
    def __init__(self):
        self.augmentation = models.Sequential(
            [
                layers.Rescaling(255),
                layers.Resizing(Config.IMAGE_SHAPE[0], Config.IMAGE_SHAPE[1]),
                utils.augment.RandomZoominAndOut((0.4, 1)),
                # layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomContrast(0.1),
                layers.RandomBrightness(0.1),
                layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1)),
                layers.Rescaling(1.0 / 255),
            ]
        )

    def __call__(self, inputs):
        return self.augmentation(inputs)


def make_square_size(
    image: np.array, target_size: int, square=False
):  # size: image size to convert
    height, width = image.shape[:2]
    ratio = target_size / max(height, width)  # ratio
    if ratio != 1:  # if sizes are not equal
        width = min(math.ceil(width * ratio), target_size)
        height = min(math.ceil(height * ratio), target_size)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # padding size
    dw = (target_size - width) / 2
    dh = (target_size - height) / 2
    top_padding = int(round(dh - 0.1))
    bottom_padding = int(round(dh + 0.1))
    left_padding = int(round(dw - 0.1))
    right_padding = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return image, ratio, (left_padding, top_padding)


class ImageLoader:
    def __init__(self, init_data=[], num_workers=4):
        self.queue = deque(init_data)
        self.num_workers = num_workers
        self.lock = threading.Lock()
        self.out_queue = []
        self.name_queue = []

    def set_queue(self, data):
        self.queue = deque(data)

    def get_image_names(self):
        return self.name_queue

    def run(self):
        self.out_queue = []
        self.name_queue = []
        threads = []
        for _ in range(self.num_workers):
            thread = threading.Thread(target=self.thread_main)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        return self.out_queue

    def thread_main(self):
        while True:
            self.lock.acquire()
            if not self.queue:
                self.lock.release()
                break
            image_path = self.queue.popleft()
            self.lock.release()

            image = self.read_image(image_path)
            self.lock.acquire()
            self.out_queue.append(image)
            self.name_queue.append(image_path)
            self.lock.release()

    def read_image(self, path):
        if not os.path.exists(path):
            id = get_filename(path)
            url = f"https://images.ygoprodeck.com/images/cards_small/{id}.jpg"
            print(f"FileNotFound: {path}")
            print(f"Try download a image from {url}")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                raise e
        return cv2.imread(path)[:, :, ::-1].copy()


class Logger:
    def __init__(self):
        now = datetime.datetime.strftime("%Y-%M-%d %H%m%s")
        self.logger = logging.basicConfig(filename=now, filemode="w")
