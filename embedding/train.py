import math
import os
import random
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from config import Config
from utils import *


def main():
    train_start = time.time()

    # data preparation
    image_pathes = common.extract_images("datasets/card_images_cropped_224")
    filenames = []
    for path in image_pathes:
        basename = os.path.basename(path)
        filenames.append(basename)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_pathes)
    image_dataset = image_dataset.map(common.load_image_to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    image_dataset = image_dataset.map(common.preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    indices = tf.data.Dataset.from_tensor_slices(list(range(len(image_dataset))))
    dataset = tf.data.Dataset.zip(image_dataset, indices)

    # initalize hyper-parameters
    model = models.EmbeddingModel(Config.IMAGE_SHAPE)
    # decay_fn = ExponentialDecay(Config.INITIAL_LEARNING_RATE, decay_steps=(batch_count) * 10, decay_rate=0.5)
    # warmup = scheduler.WarmUp(Config.INITIAL_LEARNING_RATE, decay_fn, batch_count * 3)
    optimizer = Adam(learning_rate=0.00003)

    # training
    best_loss = float("inf")
    for epoch in range(300):
        epoch_start = time.time()
        train_pos_loss = 0
        train_neg_loss = 0

        if epoch > Config.HARD_SELECT and epoch % 5 == 0:
            matrix = model.make_matrix(image_dataset, 64)

        for batch in dataset.batch(Config.BATCH_SIZE):
            anchor, indices = batch
            positive = common.augmentation(anchor)
            if epoch > Config.HARD_SELECT:
                pred_anchor = model(anchor)
                negative = common.hard_negative_selector(matrix, pred_anchor, indices, image_pathes)
            else:
                negative = common.random_negative_selector(indices, image_pathes)

            with tf.GradientTape() as tape:
                pred_anchor, pred_positive, pred_negative = model(anchor, positive, negative)
                positive_loss = losses.contrastive_loss(pred_anchor, pred_positive, 0)
                negative_loss = losses.contrastive_loss(pred_anchor, pred_negative, 1, 1)
                loss = positive_loss + negative_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_pos_loss += np.sum(positive_loss.numpy()) / len(positive_loss)
            train_neg_loss += np.sum(negative_loss.numpy()) / len(negative_loss)


        # print training loss
        epoch_time = time.time() - epoch_start
        print(f"epoch: {epoch} {datetime.now().strftime('%Y-%m-%dT %H:%M:%S')}")
        print(f"\tpositive loss: {train_pos_loss}, negative loss: {train_neg_loss}")
        print(f"\tprocessing time: {int(epoch_time) // 60}m {epoch_time % 60:.3f}s")


        # save best only
        train_loss = train_pos_loss + train_neg_loss
        if train_loss < best_loss:
            model.save_weights(f"{Config.SAVE_PATH}/ckpt_{epoch}_{train_loss:.3f}")
            best_loss = train_loss
    total_time = time.time() - train_start
    print(f"total time: {int(total_time) // 60}m {total_time % 60:.3f}s")


if __name__ == "__main__":
    main()
