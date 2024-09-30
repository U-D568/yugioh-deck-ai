import os
import sys
import cv2

from ultralytics import YOLO

sys.path.append(os.path.join(os.getcwd()))
import common

def test_detection():
    model = YOLO("weights/yolov8n_detector.pt")
    # images = common.extract_images("datasets/detector_dataset/unlabled")
    images = [cv2.imread("ancient_gear.png")]
    result = model.predict(images, save=True)


if __name__ == "__main__":
    test_detection()