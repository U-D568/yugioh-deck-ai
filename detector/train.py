import os

from ultralytics import YOLO

def train_detection():
    model = YOLO("yolov8n.pt")
    result = model.train(data="./datasets/data.yaml", batch=8)


if __name__ == "__main__":
    train_detection()