from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/processed/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="plate_yolov8n_50",
        exist_ok=True
    )


if __name__ == "__main__":
    main()