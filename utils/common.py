from collections import deque
import math
import os
import threading
import urllib

import cv2
import numpy as np

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def extract_images(path):
    images = []
    for curdir, _, files in os.walk(path):
        for file in files:
            _, ext = os.path.splitext(file)
            if not ext in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(curdir, file)
            images.append(path)
    return images


def get_filename(path):
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)
    return name


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


def is_prime(num):
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def get_factors(num):
    result = []
    limit = int(math.sqrt(num)) + 1
    for i in range(1, limit):
        if num % i == 0:
            result.append(i)
            result.append(num // i)
    return result


class ImageLoader:
    def __init__(self, init_data=[], num_workers=4):
        self.queue = deque(init_data)
        self.num_workers = num_workers
        self.lock = threading.Lock()
        self.out_queue = []
        self.name_queue = []

    def set_queue(self, data):
        self.queue = deque(data)

    def get_file_names(self):
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
