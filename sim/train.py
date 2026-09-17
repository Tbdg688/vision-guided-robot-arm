from ultralytics import YOLO

model = YOLO(r"D:\Vision_Machine\yolov8n.pt")
model.train(
    data="D:/Vision_Machine/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="cube_detector"
)