import sys

import cv2

from utils import models
from utils import common


def main():
    if len(sys.argv) < 2:
        exit(-1)

    model = models.EmbeddingModel(matrix_path="weights/matrix.npy")
    model.load("embedding.h5")

    yolo = models.YoloDetector("weights/yolov8n_detector.pt")
    input_image = cv2.imread(sys.argv[1])

    det_pred = yolo.pred([input_image])
    det_pred = det_pred[:, :, :, ::-1]

    embedding_pred = model.pred(det_pred, 32)

    card_ids = list(map(common.get_card_id_by_index, embedding_pred))
    card_ids = list(map(int, card_ids))
    card_names = list(map(common.find_card_by_id, card_ids))

    for name in card_names:
        print(name)


if __name__ == "__main__":
    main()
