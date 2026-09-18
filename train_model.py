from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=20,
    imgsz=416,
    batch=4,
    device="cpu",
    project="runs",
    name="blood_cell_model"
)

print("Training completed!")