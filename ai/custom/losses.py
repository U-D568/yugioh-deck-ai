import numpy as np
import tensorflow as tf
from tensorflow.keras import activations


def cosine_distance(x1: tf.Tensor, x2: tf.Tensor):
    inner_products = tf.tensordot(x1, x2, axes=-1)
    x1_norm = tf.norm(x1, ord="euclidean", axis=-1)
    x2_norm = tf.norm(x2, ord="euclidean", axis=-1)
    return 1 - (inner_products / (x1_norm * x2_norm))


def contrastive_loss(anchor, pred, y, margin = 0.5):
    distances = tf.math.reduce_euclidean_norm(anchor - pred, axis=-1)
    margin_distances = tf.math.reduce_euclidean_norm(anchor - pred, axis=-1)
    margin_distances = tf.maximum(margin - margin_distances, 0)
    losses = (1 - y) * tf.math.pow(distances, 2) + y * tf.math.pow(margin_distances, 2)
    return losses


def square_norm(x1, x2):
    return tf.math.reduce_sum(tf.math.square(x1 - x2), axis=-1)


def triplet_loss(anchor, positive, negative, margin=0.5):
    distance_positive = square_norm(anchor, positive)
    distance_negative = square_norm(anchor, negative)
    loss = tf.maximum(distance_positive - distance_negative + margin, 0)

    return loss