import numpy as np

from utils import *


def cal_accuracy(model, dataset, batch_size=32):
    correct = 0
    matrix = model.make_mat

    for i, batch in enumerate(dataset.batch(batch_size)):
        augmented = common.augmentation(batch)
        pred = model.predict(augmented, batch_size, verbose=False)
    
        for j, batch in enumerate(pred):
            result = losses.square_norm(matrix - pred)
            min_index = np.argmin(result.numpy())
            if min_index == i * batch_size + j:
                correct += 1
    
    accuracy = correct / len(dataset) * 100
    return accuracy
