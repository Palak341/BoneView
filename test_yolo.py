from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("ss.webp")

results[0].show()