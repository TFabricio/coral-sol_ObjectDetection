from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data="data.yaml", imgsz =640, batch=16, epochs=1000)