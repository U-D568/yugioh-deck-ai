from ultralytics import YOLO


def main():
    model = YOLO("weights/yolov8n_detector.pt")
    model.export(format="tflite", imgsz=(None, None))

if __name__ == "__main__":
    main()