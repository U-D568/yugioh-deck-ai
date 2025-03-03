import tensorflow as tf
import numpy as np

import utils

# convert to tflite

model = utils.models.EmbeddingModel()
model.load_weights("weights/effnet_checkpoints/ckpt_355_0.990")

tf.saved_model.save(model.model, "weights/embeddingModel")
loaded_model = tf.saved_model.load("weights/embeddingModel")

converter = tf.lite.TFLiteConverter.from_saved_model("weights/embeddingModel")
tflite_model = converter.convert()

with open("weights/model.tflite", "w+b") as fp:
    fp.write(tflite_model)



# interpreter = tf.lite.Interpreter(model_path="weights/model.tflite")
# interpreter.resize_tensor_input(0, [3, 224, 224, 3])
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# print("Input Shape:", input_details[0]["shape_signature"])

# output_details = interpreter.get_output_details()
# print("Output Shape:", output_details[0]["shape_signature"])

# input_data = np.array(np.random.random_sample([3, 224, 224, 3]), dtype=np.float32)
# input_tensor = interpreter.set_tensor(input_details[0]['index'], input_data)

# interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)