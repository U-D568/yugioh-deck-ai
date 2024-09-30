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
import custom
import common


def main():
    # Constants
    TOTAL_START = time.time()


    # data preparation
    image_pathes = common.extract_images(Config.IMAGE_PATH)
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
    batch_count = len(dataset.batch(Config.BATCH_SIZE))
    model = custom.models.EmbeddingModel(Config.IMAGE_SHAPE)
    # model.load_weights("./effnet_checkpoints/ckpt_298_1.575")
    # decay_fn = ExponentialDecay(Config.INITIAL_LEARNING_RATE, decay_steps=(batch_count) * 10, decay_rate=0.5)
    # warmup = custom.scheduler.WarmUp(Config.INITIAL_LEARNING_RATE, decay_fn, batch_count * 3)
    optimizer = Adam(learning_rate=0.00003)

    # training
    best_loss = float("inf")
    for epoch in range(299, 500):
        
        epoch_start = time.time()
        train_pos_loss = 0
        train_neg_loss = 0

        # if (epoch + 1) % Config.OFFLINE_SELECT == 0:
        #     matrix = model.make_matrix(image_dataset, 64)

        for batch in dataset.batch(Config.BATCH_SIZE):
            anchor, indices = batch
            positive = common.augmentation(anchor)
            negative = common.random_negative_selector(indices, image_pathes)
            # if epoch > Config.HARD_SELECT:
            #     pred_anchor = model(anchor)
            #     negative = common.hard_negative_selector(matrix, pred_anchor, indices, image_pathes)

            with tf.GradientTape() as tape:
                pred_anchor, pred_positive, pred_negative = model(anchor, positive, negative)
                positive_loss = custom.losses.contrastive_loss(pred_anchor, pred_positive, 0)
                negative_loss = custom.losses.contrastive_loss(pred_anchor, pred_negative, 1, 1)
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
    total_time = time.time() - TOTAL_START
    print(f"total time: {int(total_time) // 60}m {total_time % 60:.3f}s")


if __name__ == "__main__":
    main()
