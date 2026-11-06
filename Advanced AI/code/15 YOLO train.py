import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict
import yaml
from datetime import datetime
import random

class Config:
    BASE_DIR = "dataset/KolektorSDD_resized"
    SAVE_DIR = "Yolo"
    LOG_DIR = os.path.join(SAVE_DIR, "../../asr/whisper-runs/Yolo")
    IMG_SIZE = 1024
    EPOCHS = 100
    BATCH_SIZE = 8
    VAL_RATIO = 0.2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    USE_BBOX = True


def set_seed(seed: int = Config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DatasetValidator:
    @staticmethod
    def validate_structure(BASE_DIR: str, folders: List[str]) -> None:
        missing = [f for f in folders if not os.path.exists(os.path.join(BASE_DIR, f))]
        if missing:
            raise FileNotFoundError(f"Missing folders: {missing}")

    @staticmethod
    def validate_pairs(BASE_DIR: str, folders: List[str]) -> None:
        missing = []
        for folder in folders:
            for i in range(8):
                img = os.path.join(BASE_DIR, folder, f"Part{i}.jpg")
                mask = os.path.join(BASE_DIR, folder, f"Part{i}_label.bmp")
                if not os.path.exists(img) or not os.path.exists(mask):
                    missing.append((folder, i))
        if missing:
            raise FileNotFoundError(f"Missing image-mask pairs: {missing}")


class YOLODatasetPreparer:
    def __init__(self, config: Config):
        self.config = config
        set_seed(config.SEED)
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

    def create_structure(self) -> str:
        yolo_dir = os.path.join(self.config.SAVE_DIR, 'yolo_data')
        for d in ['images', 'labels']:
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(yolo_dir, d, split), exist_ok=True)
        return yolo_dir

    def mask_to_yolo_label(self, mask_path: str, label_path: str) -> None:
        mask = np.array(Image.open(mask_path).convert('L'))
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(label_path, 'w') as f:
            for contour in contours:
                if self.config.USE_BBOX:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = (x + w / 2) / mask.shape[1], (y + h / 2) / mask.shape[0]
                    nw, nh = w / mask.shape[1], h / mask.shape[0]
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                else:
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 6:
                        points = approx.squeeze() / np.array([mask.shape[1], mask.shape[0]])
                        f.write("0 " + " ".join(map(str, points.flatten())) + "\n")

    def prepare_dataset(self, yolo_dir: str):
        folders = [f"kos{str(i).zfill(2)}" for i in range(1, 51)]
        DatasetValidator.validate_structure(self.config.BASE_DIR, folders)
        DatasetValidator.validate_pairs(self.config.BASE_DIR, folders)

        train_val, test = train_test_split(folders, test_size=0.1, random_state=self.config.SEED)
        train, val = train_test_split(train_val, test_size=self.config.VAL_RATIO, random_state=self.config.SEED)

        for split, items in [('train', train), ('val', val), ('test', test)]:
            for folder in items:
                for i in range(8):
                    img_src = os.path.join(self.config.BASE_DIR, folder, f"Part{i}.jpg")
                    mask_src = os.path.join(self.config.BASE_DIR, folder, f"Part{i}_label.bmp")
                    img_dst = os.path.join(yolo_dir, 'images', split, f"{folder}_Part{i}.jpg")
                    label_dst = os.path.join(yolo_dir, 'labels', split, f"{folder}_Part{i}.txt")
                    shutil.copy2(img_src, img_dst)
                    self.mask_to_yolo_label(mask_src, label_dst)

        with open(os.path.join(yolo_dir, 'data.yaml'), 'w') as f:
            yaml.dump({
                'path': yolo_dir,
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': 1,
                'names': ['crack']
            }, f)


class YOLOTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, datetime.now().strftime("%Y%m%d_%H%M%S")))

    def train_model(self, data_yaml: str) -> YOLO:
        model = YOLO('yolov8m.pt')
        model.train(
            data=data_yaml,
            epochs=self.config.EPOCHS,
            batch=self.config.BATCH_SIZE,
            imgsz=self.config.IMG_SIZE,
            device=self.config.DEVICE,
            optimizer='AdamW',
            lr0=0.001,
            weight_decay=0.0005,
            dropout=0.1,
            name='yolo_crack_model',
            project=self.config.SAVE_DIR
        )
        return model

    def log_metrics(self, model: YOLO):
        results_csv = os.path.join(self.config.SAVE_DIR, 'yolo_crack_model', 'results.csv')
        if not os.path.exists(results_csv):
            print("Results CSV not found.")
            return

        df = pd.read_csv(results_csv)
        for col in df.columns:
            if any(k in col for k in ['loss', 'mAP', 'precision', 'recall']):
                for epoch, val in enumerate(df[col].dropna()):
                    self.writer.add_scalar(col, val, epoch)

    def evaluate_model(self, model: YOLO, data_yaml: str) -> Dict:
        metrics = model.val(data=data_yaml, split='test')
        return {
            'map50': metrics.box.map50(),
            'map': metrics.box.map(),
            'precision': metrics.box.mp(),
            'recall': metrics.box.mr()
    }


    def export_model(self, model: YOLO):
        export_path = os.path.join(self.config.SAVE_DIR, 'exported')
        os.makedirs(export_path, exist_ok=True)
        model.export(format='onnx', dynamic=True, simplify=True, opset=12, name='crack_model.onnx', project=export_path)
        if torch.cuda.is_available():
            model.export(format='engine', name='crack_model.engine', project=export_path)


def main():
    set_seed(Config.SEED)
    preparer = YOLODatasetPreparer(Config)
    yolo_dir = preparer.create_structure()
    preparer.prepare_dataset(yolo_dir)
    data_yaml = os.path.join(yolo_dir, 'data.yaml')

    trainer = YOLOTrainer(Config)
    model = trainer.train_model(data_yaml)
    trainer.log_metrics(model)
    metrics = trainer.evaluate_model(model, data_yaml)
    print("Evaluation Metrics:", metrics)
    trainer.export_model(model)
    model.save(os.path.join(Config.SAVE_DIR, 'best_yolo_crack_model.pt'))


if __name__ == "__main__":
    main()