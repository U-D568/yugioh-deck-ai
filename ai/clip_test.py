import tensorflow as tf
from tensorflow.keras.activations import relu


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        return self.dense(inputs)

def mse(pred, y):
    return tf.reduce_sum(tf.square(pred - y), axis=-1)

def mse_clip(pred, y):
    loss = tf.reduce_sum(tf.square(pred - y), axis=-1)
    return tf.maximum(loss, 1)

def mse_clip_grad(pred, y):
    loss = tf.reduce_sum(tf.square(pred - y), axis=-1)
    return relu(loss, max_value=2)

def main():
    x = tf.constant([[1], [1], [1]], dtype=tf.float32)
    y = tf.constant([[1], [1], [1]], dtype=tf.float32)
    model = SimpleModel()

    with tf.GradientTape() as tape:
        pred = model(x)
        loss = mse_clip(pred, y) + 100
        
    gradients = tape.gradient(loss, model.trainable_variables)
    for grad in gradients:
        print(grad.numpy())


if __name__ == "__main__":
    main()