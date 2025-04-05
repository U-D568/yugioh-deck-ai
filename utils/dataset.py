from collections.abc import Sequence
import math
import random

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset

# test
if __name__ == "__main__":
    import sys, os
    sys.path.append(f"{os.getcwd()}")

from utils import common, augment, losses


class DecklistDataset(Dataset):
    def __init__(self, image_pathes, is_pendulum, deck_shape=60, num_workers=4):
        self.image_loader = common.ImageLoader(num_workers=num_workers)
        self.device = torch.device
        self.card_data = list(zip(image_pathes, is_pendulum))
        self.deck_shape = deck_shape

        # Each element is number of cards in a image
        self.deck_sizes = self.make_deck_sizes()
        self.deck_data = self.make_deck_images()

        # relative coordinates of ilust in image
        self.card_size = np.array([268, 391])
        self.normal_pos = np.array([32, 72, 236, 275]) / np.tile(self.card_size, 2)
        self.pendulum_pos = np.array([17, 70, 250, 242]) / np.tile(self.card_size, 2)
    
    def make_deck_images(self):
        i = 0
        deck_images = []
        random.shuffle(self.card_data)
        for size in self.deck_sizes:
            deck_images.append(self.card_data[i : i + size])
            i = min(i + size, len(self.card_data) - 1)

        return deck_images
    
    def make_deck_sizes(self):
        if isinstance(self.deck_shape, Sequence) and all(n for n in self.deck_shape):
            deck_sizes = self.make_deck_size_in_range(self.deck_shape)
        elif isinstance(self.deck_shape, int):
            deck_sizes = self.make_fixed_deck_size(self.deck_shape)
        else:
            raise ValueError(f"Invalid deck_shape value. It should be int or sequence of int. used value: {self.deck_shape}")

        random.shuffle(deck_sizes)
        return deck_sizes

    def make_deck_size_in_range(self, deck_size_range):
        data_length = len(self.card_data)
        deck_size_list = []

        while data_length > 0:
            num = random.randint(deck_size_range[0], deck_size_range[1])
            if data_length >= num:
                deck_size_list.append(num)
                data_length -= num
            else:
                deck_size_list.append(data_length)
                data_length = 0

        random.shuffle(deck_size_list)
        return deck_size_list

    def make_fixed_deck_size(self, deck_size):
        data_length = len(self.card_data)
        div = data_length // deck_size
        mod = data_length % deck_size

        deck_size_list = [deck_size] * div
        if mod > 0:
            deck_size_list.append(mod)

        random.shuffle(deck_size_list)
        return deck_size_list

    def __len__(self):
        return len(self.deck_sizes)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        image_pathes = [data[0] for data in self.deck_data[index]]
        is_pendulum = [data[1] for data in self.deck_data[index]]

        self.image_loader.set_queue(image_pathes)
        images = self.image_loader.run()
        ids = self.image_loader.get_file_names()
        ids = list(map(common.get_filename, ids))

        xyxy = []
        for i, image in enumerate(images):
            image, ratio, offset = augment.zoomout_and_transition(image)
            h, w, _ = image.shape
            images[i] = image
            pos = self.pendulum_pos.copy() if is_pendulum[i] else self.normal_pos.copy()
            pos = pos * ratio * np.array([w, h, w, h])
            pos += np.tile(offset, 2)
            pos = np.round(pos)
            xyxy.append(pos)
        xyxy = np.stack(xyxy)

        origianl_image, offset = augment.make_deck_image(images)
        xyxy += np.tile(offset, 2)

        deck_image, ratio, offset = common.make_square_size(origianl_image, 640)
        height, width, _ = deck_image.shape
        # card image position(xyxy)
        xyxy = xyxy * ratio + np.tile(offset, 2)
        xyxy[:, [0, 2]] /= width
        xyxy[:, [1, 3]] /= height

        # xyxy to xywh
        # shape -> (60, 4)
        width = xyxy[:, 2] - xyxy[:, 0]
        height = xyxy[:, 3] - xyxy[:, 1]
        xywh = xyxy.copy()
        xywh[:, 0] += width / 2
        xywh[:, 1] += height / 2
        xywh[:, 2] = width
        xywh[:, 3] = height

        return {
            "image": deck_image,
            "ids": ids,
            "xyxy": xyxy,
            "xywh": xywh,
            "ori_image": origianl_image,
        }

    def collate_fn(self, data_list):
        batch = {"batch_idx": []}

        for i, data in enumerate(data_list):
            for key in data:
                values = batch.get(key, [])
                values.append(data[key])
                batch[key] = values
            batch["batch_idx"].append(np.array([i] * len(data["xyxy"])))
        batch["xywh"] = np.concatenate(batch["xywh"], axis=0)
        batch["xyxy"] = np.concatenate(batch["xyxy"], axis=0)
        batch["batch_idx"] = np.concatenate(batch["batch_idx"], axis=0)
        batch["image"] = np.stack(batch["image"], axis=0)

        return batch

    def shuffle(self):
        self.deck_sizes = self.make_deck_sizes()
        self.deck_data = self.make_deck_images()

    def preprocessing(self, image):
        image = image[:, :, ::-1]  # bgr -> rgb
        image = torch.from_numpy(image).float()
        image = torch.permute(image, [2, 0, 1])  # hwc -> chw
        return image


class EmbeddingDataset:
    def __init__(self, path):
        df = pd.read_csv(path)
        self.length = len(df)
        self.img_path = df["id"].tolist()
        self.img_path = list(
            map(
                lambda x: "datasets/card_images_small/" + str(x) + ".jpg", self.img_path
            )
        )
        self.is_pendulum = list(
            map(lambda x: x.lower().startswith("pendulum"), df["type"].tolist())
        )

        self.input_size = (224, 224)
        self.image_size = [391, 268]  # height, width
        self.normal_pos = tf.convert_to_tensor([[72, 32, 275, 236]]) / tf.tile(
            self.image_size, [2]
        )  # y1, x1, y2, x2 (normalized)
        self.normal_pos = tf.cast(self.normal_pos, dtype=tf.float32)
        self.pendulum_pos = tf.convert_to_tensor([[70, 17, 242, 250]]) / tf.tile(
            self.image_size, [2]
        )
        self.pendulum_pos = tf.cast(self.pendulum_pos, dtype=tf.float32)

        self.dataset = self._build_dataset()

    def __add__(self, other):
        self.img_path = self.img_path + other.img_path
        self.is_pendulum = self.is_pendulum + other.is_pendulum
        return self

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self._load_image(self.img_path[index])
        return self._preprocess(img, self.is_pendulum[index])

    def _generator(self):
        for path, is_pendulum in iter(zip(self.img_path, self.is_pendulum)):
            yield path, is_pendulum

    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def _preprocess(self, image, is_pendulum):
        crop_box = self.pendulum_pos if is_pendulum else self.normal_pos
        crop_img = tf.image.crop_and_resize(
            image=tf.expand_dims(image, axis=0),
            boxes=crop_box,
            box_indices=[0],
            crop_size=self.input_size,
            method="bilinear",
        )
        return tf.squeeze(crop_img, axis=0) / 255.0

    def _build_dataset(self):
        img_dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.bool),
            ),
        )
        img_dataset = img_dataset.map(
            lambda path, is_pendulum: self._preprocess(
                self._load_image(path), is_pendulum
            )
        )
        index = tf.data.Dataset.from_tensor_slices(list(range(self.length)))
        dataset = tf.data.Dataset.zip(img_dataset, index)
        return dataset


class EmbeddingMatrix:
    def __init__(self, embedding_model, dataset):
        self.model = embedding_model
        self.dataset = dataset
        self.matrix = None

    def update_matrix(self, batch_size=32):
        matrix = []
        for batch in self.dataset.dataset.batch(batch_size):
            img, _ = batch
            embeds = self.model.predict(img, batch_size=batch_size)
            matrix.append(embeds)
        self.matrix = tf.concat(matrix, axis=0)

    def get_hard_negative(self, anchor, anchor_index):
        taraget_index = -1
        min_distance = math.inf

        if anchor_index > 0:
            lower_matrix = self.matrix[:anchor_index]
            lower_matrix = losses.square_norm(lower_matrix, anchor)
            min_index = tf.argmin(lower_matrix, axis=0, output_type=tf.int32)
            distance = lower_matrix[min_index]
            if distance < min_distance:
                min_distance = distance
                taraget_index = min_index

        if anchor_index < self.dataset.length - 1:
            upper_matrix = self.matrix[anchor_index + 1 :]
            upper_matrix = losses.square_norm(upper_matrix, anchor)
            min_index = tf.argmin(upper_matrix, axis=0, output_type=tf.int32)
            distance = upper_matrix[min_index]
            if distance < min_distance:
                min_distance = distance
                taraget_index = min_index

        return taraget_index

    def get_random_negative(self, anchor_index):
        target_index = -1
        while True:
            target_index = random.randrange(len(self.dataset))
            if target_index != anchor_index:
                break
        return target_index


if __name__ == "__main__":
    def make_dataset(df_path):
        X_train = pd.read_csv(df_path)
        id_list = X_train["id"].tolist()
        prefix = "datasets/card_images_small/"
        id_list = list(map(lambda x: prefix + str(x) + ".jpg", id_list))
        card_type = list(
            map(lambda x: x.lower().startswith("pendulum"), X_train["type"].tolist())
        )
        return DecklistDataset(id_list, card_type, (1, 5))

    valid_dataset = make_dataset("datasets/valid.csv")
    for i in range(20):
        valid_dataset[i]
    print(1)
