from concurrent import futures
import math
import os
import random
from typing import List
import urllib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import common, augment


class DecklistDataset(Dataset):
    def __init__(self, image_pathes, is_pendulum, deck_size=60, num_workers=4):
        self.device = torch.device
        self.indexes = list(range(len(image_pathes)))
        self.image_pathes = image_pathes
        self.is_pendulum = is_pendulum
        self.deck_size = deck_size
        self.image_loader = common.ImageLoader(num_workers=num_workers)
        # self.transform = transforms.ColorJitter(
        #     brightness=0.25, saturation=0.5, contrast=0.25
        # )

        # card ilustration relative coordinates
        self.card_size = np.array([268, 391])
        self.normal_pos = np.array([32, 72, 236, 275]) / np.tile(self.card_size, 2)
        self.pendulum_pos = np.array([17, 70, 250, 242]) / np.tile(self.card_size, 2)
        self.shuffle()

    def __len__(self):
        return math.ceil((len(self.image_pathes) / self.deck_size))

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        chunk = self.indexes[index * self.deck_size : (index + 1) * self.deck_size]
        image_pathes = [self.image_pathes[i] for i in chunk]
        is_pendulum = [self.is_pendulum[i] for i in chunk]

        self.image_loader.set_queue(image_pathes)
        images = self.image_loader.run()
        ids = self.image_loader.get_image_names()
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
        if len(xyxy) == 0:
            print(1)
            return None, None, None
        xyxy = np.stack(xyxy)

        origianl_image, offset = augment.make_deck_image(images)
        xyxy += np.tile(offset, 2)
        # Debug
        # for xyxy in ilust_xyxy:
        #     cv2.rectangle(deck_image, xyxy[:2].astype(np.int32), xyxy[2:].astype(np.int32), (255, 0, 0), 1)
        # cv2.imwrite("test.png", deck_image[:,:,::-1])

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

        # Debug
        # for xyxy in ilust_xyxy:
        #     cv2.rectangle(deck_image, xyxy[:2].astype(np.int32), xyxy[2:].astype(np.int32), (255, 0, 0), 1)
        # cv2.imwrite("test2.png", deck_image[:,:,::-1])

        return {"image": deck_image, "ids": ids, "xyxy": xyxy, "xywh": xywh, "ori_image": origianl_image}

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
        random.shuffle(self.indexes)

    def preprocessing(self, image):
        image = image[:, :, ::-1]  # bgr -> rgb
        image = torch.from_numpy(image).float()
        image = torch.permute(image, [2, 0, 1])  # hwc -> chw
        return image
