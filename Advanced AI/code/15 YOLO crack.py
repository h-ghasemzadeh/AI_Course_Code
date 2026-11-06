from ultralytics import YOLO

# بارگذاری مدل
model = YOLO("yolov8n.pt")

# اجرا روی تصویر ترک
results = model("steel_crack.jpg", save=True)
