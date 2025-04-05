import os
import sys
import time
from datetime import datetime
import cv2
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

sys.path.append(f"{os.getcwd()}")

from utils import common, models, losses, logger
from utils.dataset import EmbeddingDataset, EmbeddingMatrix


def main():
    log = logger.TrainLogger()

    # train hyper parameter
    HARD_SELECT = 0
    BATCH_SIZE = 8
    INPUT_SHAPE = (224, 224, 3)
    SAVE_PATH = "embedding/weights"

    # preprocessor
    image_preprocess = common.EmbeddingPreprocessor()
    augmentation = common.EmbeddingAugmentation()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # initalize hyper-parameters
    model = models.EmbeddingModel(INPUT_SHAPE)
    model.load("embedding/weights/last.h5")

    # data preparation
    train_dataset = EmbeddingDataset("datasets/train.csv")
    valid_dataset = EmbeddingDataset("datasets/valid.csv")
    # valid_dataset = valid_dataset + train_dataset
    valid_matrix = EmbeddingMatrix(model, valid_dataset)
    valid_matrix.update_matrix()
    gc.collect()

    # training
    hit_count = 0
    false_data = []
    false_pred = []
    for batch in valid_dataset.dataset.batch(BATCH_SIZE):
        anchor_img, index = batch
        positive_img = augmentation(anchor_img)
        
        pred_positive = model(positive_img)
        result = losses.cosine_distance(valid_matrix.matrix[None, :], pred_positive[:, None, :])
        pred_index = tf.argmin(result, axis=1, output_type=tf.int32)
        hit = tf.math.equal(index, pred_index)
        hit_count += tf.math.count_nonzero(hit).numpy()
        for i, h in enumerate(hit):
            if h == True:
                continue
            false_data.append(index[i])
            false_pred.append(pred_index[i])

    
    print(f"hit count: {hit_count} out of {len(valid_dataset)}")
    print(f"accuracy: {hit_count / len(valid_dataset) * 100}%")



if __name__ == "__main__":
    main()
