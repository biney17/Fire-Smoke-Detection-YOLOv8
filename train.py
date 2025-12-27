from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # bon équilibre vitesse/précision

    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    main()
