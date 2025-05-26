model = YOLO("yolo11m-obb.pt")
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=(1024, 1024),
    # task="detection"
    device=0
)
