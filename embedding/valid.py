import os
import sys
import gc

import tensorflow as tf

sys.path.append(f"{os.getcwd()}")
from data.dataset.tf import EmbeddingDataset
from data.augmentation.tf import EmbeddingAugmentation
from data.preprocess.tf import EmbeddingPreprocessor
from structures import EmbeddingMatrix
from loss.tf import cosine_distance
from models.tf import EmbeddingModel


try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except RuntimeError as e:
    raise e


def main():
    # train hyper parameter
    HARD_SELECT = 0
    BATCH_SIZE = 8
    INPUT_SHAPE = (224, 224, 3)

    # preprocessor
    image_preprocess = EmbeddingPreprocessor()
    augmentation = EmbeddingAugmentation(0.1, 0.4)

    # initalize hyper-parameters
    model = EmbeddingModel(INPUT_SHAPE)
    model.load("embedding/weights/best.h5")

    # data preparation
    train_dataset = EmbeddingDataset.load("datasets/train.csv")
    valid_dataset = EmbeddingDataset.load("datasets/valid.csv")
    valid_dataset = valid_dataset + train_dataset
    valid_matrix = EmbeddingMatrix(model, valid_dataset)
    valid_matrix.update_matrix()

    valid_dataset = EmbeddingDataset.load("datasets/valid.csv")
    gc.collect()

    hit_count = 0
    false_data = []
    false_pred = []
    for batch in valid_dataset.dataset.batch(BATCH_SIZE):
        anchor_img, index = batch
        positive_img = augmentation(anchor_img)

        pred_positive = model(positive_img)
        result = cosine_distance(
            valid_matrix.matrix[None, :], pred_positive[:, None, :]
        )
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
    print(1)


if __name__ == "__main__":
    main()
