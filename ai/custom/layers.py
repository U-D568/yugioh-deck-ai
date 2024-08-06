import tensorflow as tf
from tensorflow.keras import layers

class DistanceLayer(layers.Layer):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
    
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1) # anchor-positive distance
        an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1) # anchor-negative distance
        return ap_distance, an_distance