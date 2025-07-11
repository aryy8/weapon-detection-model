from ultralytics import YOLO

model = YOLO("best.pt")
results = model.val(data="/Users/aryan/WeaponDetection/data.yaml")
