import math
import os
import random
import time

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
    image_dataset = image_dataset.map(
        common.load_image_to_tensor, num_parallel_calls=tf.data.AUTOTUNE
    )
    image_dataset = image_dataset.map(
        common.preprocessing, num_parallel_calls=tf.data.AUTOTUNE
    )
    indices = tf.data.Dataset.from_tensor_slices(list(range(len(image_dataset))))
    dataset = tf.data.Dataset.zip(image_dataset, indices)

    # training hyper-parameters
    batch_count = len(dataset.batch(Config.BATCH_SIZE))

    model = custom.models.SiameseModel()
    model.load_weights(f"ai/checkpoints/ckpt_{Config.CHECKPOINT}")

    decay_fn = ExponentialDecay(
        Config.INITIAL_LEARNING_RATE, decay_steps=(batch_count) * 10, decay_rate=0.5
    )
    # warmup = custom.scheduler.WarmUp(
    #     Config.INITIAL_LEARNING_RATE, decay_fn, batch_count * 3
    # )
    optimizer = Adam(decay_fn)

    # training
    best_loss = None
    matrix = common.make_matrix(model, image_dataset, 32)
    for epoch in range(Config.CHECKPOINT + 1, Config.EPOCHS):
        epoch_start = time.time()
        triplet_loss = 0
        pos_train_loss = 0
        neg_train_loss = 0

        if (epoch + 1) % Config.OFFLINE_SELECT == 0:
            common.make_matrix(model, image_dataset, 32)

        print(f"epoch: {epoch}")
        for batch in dataset.batch(Config.BATCH_SIZE):
            anchor, indices = batch
            positive = common.augmentation(tf.identity(anchor))

            with tf.GradientTape() as tape:
                anchor_pred = model(anchor)
                positive_pred = model(positive)
                positive_loss = custom.losses.contrastive_loss(anchor_pred, positive_pred, 0, 0.5)
            gradients = tape.gradient(positive_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            pos_train_loss += np.sum(positive_loss.numpy()) / len(positive_loss)

            with tf.GradientTape() as tape:
                anchor_pred = model(anchor)
                negative = common.hard_negative_selector(matrix, anchor_pred, indices, image_pathes)
                negative_pred = model(negative)
                negative_loss = custom.losses.contrastive_loss(anchor_pred, negative_pred, 1, 0.5)
            gradients = tape.gradient(negative_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            neg_train_loss += np.sum(negative_loss.numpy()) / len(negative_loss)

                # losses = custom.losses.triplet_loss(
                #     anchor_pred, positive_pred, negative_pred, margin=1
                # )

            # gradients = tape.gradient(losses, model.trainable_variables)
            # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # triplet_loss += np.sum(losses.numpy())

        # print training loss
        triplet_loss /= batch_count
        epoch_time = time.time() - epoch_start
        print(f"\ttriplet loss: {triplet_loss}")
        print(f"\tprocessing time: {int(epoch_time) // 60}m {epoch_time % 60:.3f}s")

        # save best only
        if best_loss is None or triplet_loss < best_loss:
            model.save_weights(f"ai/checkpoints/ckpt_{epoch}")
            best_loss = triplet_loss
    total_time = time.time() - TOTAL_START
    print(f"total time: {int(total_time) // 60}m {total_time % 60:.3f}s")


if __name__ == "__main__":
    main()
