import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# تنظیمات مدل از پیش آموزش دیده‌شده Fast R-CNN (ResNet-50)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # آستانه اعتماد برای تشخیص‌ها
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# 📷 خواندن تصویر ورق فولادی دارای ترک
image = cv2.imread("steel_crack.jpg")  # تصویر باید در مسیر پروژه باشد

# 🔍 تشخیص
outputs = predictor(image)

# 🖼️ نمایش نتایج با جعبه دور ترک‌ها
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# نمایش تصویر نهایی
cv2.imshow("Detected Cracks", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
