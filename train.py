from ultralytics import YOLO

print("Starting improved training...")

# Use a slightly bigger model (better accuracy than yolov8n)
model = YOLO("yolov8s.pt")

model.train(
    data="dataset/data.yaml",
    epochs=40,        # more training
    imgsz=640,
    batch=8,          # adjust if memory error
    lr0=0.001,        # better learning rate
    patience=10,      # early stopping
    augment=True      # data augmentation ON
)