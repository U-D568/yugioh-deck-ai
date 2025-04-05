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
    HARD_SELECT = 20
    BATCH_SIZE = 8
    INPUT_SHAPE = (224, 224, 3)
    SAVE_PATH = "embedding/weights"

    # preprocessor
    image_preprocess = common.EmbeddingPreprocessor()
    augmentation = common.EmbeddingAugmentation()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    train_start = time.time()

    # initalize hyper-parameters
    model = models.EmbeddingModel(INPUT_SHAPE)
    model.load("embedding/weights/last.h5")
    optimizer = Adam(learning_rate=1e-5)

    # data preparation
    train_dataset = EmbeddingDataset("datasets/train.csv")
    # valid_dataset = EmbeddingDataset("datasets/valid.csv")
    train_matrix = EmbeddingMatrix(model, train_dataset)
    gc.collect()

    # training
    best_loss = float("inf")
    for epoch in range(50, 300):
        epoch_start = time.time()
        train_loss = 0

        if epoch >= HARD_SELECT and epoch % 5 == 0:
            train_matrix.update_matrix()

        for batch in train_dataset.dataset.batch(BATCH_SIZE):
            anchor_img, indices = batch
            batch_size = anchor_img.shape[0]
            positive_img = augmentation(anchor_img)
            pred_anchor = model(anchor_img)

            # select negative data
            negative_index = []
            if epoch >= HARD_SELECT:
                for anchor, anchor_index in zip(pred_anchor, indices):
                    negative_index.append(
                        train_matrix.get_hard_negative(anchor, anchor_index)
                    )
            else:
                for anchor_index in indices:
                    negative_index.append(
                        train_matrix.get_random_negative(anchor_index)
                    )

            negative_img = []
            for index in negative_index:
                img = train_dataset[index]
                negative_img.append(img)
            negative_img = tf.stack(negative_img, axis=0)
            negative_img = augmentation(negative_img)

            with tf.GradientTape() as tape:
                pred_anchor = model(anchor_img)
                pred_positive = model(positive_img)
                pred_negative = model(negative_img)
                pos_loss = losses.contrastive_loss(pred_anchor, pred_positive, 0, 0.5)
                neg_loss = losses.contrastive_loss(pred_anchor, pred_negative, 1, 0.5)
                loss = pos_loss + neg_loss
            gradients = tape.gradient(loss, model.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

            train_loss += np.sum(loss.numpy()) / batch_size

        gc.collect()
        # logging
        epoch_time = time.time() - epoch_start
        log.info(f"epoch: {epoch} {datetime.now().strftime('%Y-%m-%dT %H:%M:%S')}")
        log.info(f"\ttrain loss: {train_loss}")
        log.info(f"\tprocessing time: {int(epoch_time) // 60}m {epoch_time % 60:.3f}s")

        # save best only
        train_loss
        if train_loss < best_loss:
            model.save("embedding/weights/best.h5")
            best_loss = train_loss
        model.save("embedding/weights/last.h5")
        gc.collect()

    total_time = time.time() - train_start
    log.info(f"total time: {int(total_time) // 60}m {total_time % 60:.3f}s")


if __name__ == "__main__":
    main()
