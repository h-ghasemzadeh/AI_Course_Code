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
from typing import List, Tuple, Dict, Optional
import yaml
from datetime import datetime
import random

# Configuration Management with Hydra (commented as optional)
# import hydra
# from hydra.core.config_store import ConfigStore
# from dataclasses import dataclass

# @dataclass
# class AppConfig:
#     BASE_DIR: str = "D:/research/project/AI steel crack/dataset/KolektorSDD_resized"
#     save_dir: str = "C:/lecture/AI/support/code/Yolo/yolo_training"
#     img_size: int = 640
#     epochs: int = 50
#     batch_size: int = 8
#     val_ratio: float = 0.2
#     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
#     seed: int = 42

# cs = ConfigStore.instance()
# cs.store(name="config", node=AppConfig)

class Config:
    """Configuration class with default values"""
    BASE_DIR = r"dataset/KolektorSDD_resized"
    SAVE_DIR = r"Yolo"
    LOG_DIR = os.path.join(SAVE_DIR, "../asr/whisper-runs/Yolo")  # Directory for TensorBoard logs
    IMG_SIZE = 640
    EPOCHS = 50
    BATCH_SIZE = 8
    VAL_RATIO = 0.2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    AUGMENTATION = {
        'hsv_h': 0.2,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'flipud': 0.5,
        'translate': 0.1,
        'scale': 0.5,
    }

def set_seed(seed: int = Config.SEED) -> None:
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DatasetValidator:
    @staticmethod
    def validate_folder_structure(BASE_DIR: str, expected_folders: List[str]) -> None:
        """Validate that all expected folders exist"""
        missing = [f for f in expected_folders if not os.path.exists(os.path.join(BASE_DIR, f))]
        if missing:
            raise FileNotFoundError(f"Missing required folders: {missing}")

    @staticmethod
    def validate_image_mask_pairs(BASE_DIR: str, folders: List[str]) -> None:
        """Validate that all images have corresponding masks"""
        missing_pairs = []
        for folder in folders:
            folder_path = os.path.join(BASE_DIR, folder)
            for i in range(8):  # Assuming 8 parts per folder
                img_path = os.path.join(folder_path, f"Part{i}.jpg")
                mask_path = os.path.join(folder_path, f"Part{i}_label.bmp")
                
                if not all(os.path.exists(p) for p in [img_path, mask_path]):
                    missing_pairs.append((folder, i))
        
        if missing_pairs:
            raise FileNotFoundError(f"Missing image-mask pairs in folders: {missing_pairs}")

class YOLODatasetPreparer:
    def __init__(self, config: Config):
        self.config = config
        set_seed(self.config.SEED)
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        os.makedirs(self.config.LOG_DIR, exist_ok=True)

    def create_yolo_structure(self) -> str:
        """Create YOLO directory structure with error handling"""
        yolo_dir = os.path.join(self.config.SAVE_DIR, 'yolo_data')
        try:
            for dir_type in ['images', 'labels']:
                for split in ['train', 'val', 'test']:
                    os.makedirs(os.path.join(yolo_dir, dir_type, split), exist_ok=True)
            return yolo_dir
        except Exception as e:
            raise RuntimeError(f"Error creating directory structure: {e}")

    @staticmethod
    def mask_to_yolo_label(mask_path: str, label_path: str) -> None:
        """Convert mask to YOLO format with validation and contour simplification"""
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            with open(label_path, 'w') as f:
                for contour in contours:
                    # Simplify contour with adaptive epsilon
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    height, width = mask.shape
                    points = approx.squeeze()
                    
                    # Normalize coordinates and ensure minimum 6 points
                    if len(points) >= 6:
                        points = points / np.array([width, height])
                        points = points.flatten().tolist()
                        line = '0 ' + ' '.join(map(str, points)) + '\n'
                        f.write(line)
        except Exception as e:
            raise RuntimeError(f"Error processing mask {mask_path}: {e}")

    def prepare_dataset(self, yolo_dir: str) -> None:
        """Prepare dataset with train/val/test split and comprehensive validation"""
        all_folders = [f"kos{str(i).zfill(2)}" for i in range(1, 51)]
        
        # Validate dataset structure first
        DatasetValidator.validate_folder_structure(self.config.BASE_DIR, all_folders)
        DatasetValidator.validate_image_mask_pairs(self.config.BASE_DIR, all_folders)

        # Split into train/val/test
        train_val_folders, test_folders = train_test_split(
            all_folders, test_size=0.1, random_state=self.config.SEED
        )
        train_folders, val_folders = train_test_split(
            train_val_folders, test_size=self.config.VAL_RATIO, random_state=self.config.SEED
        )

        # Process each split
        for split, folders in [('train', train_folders), 
                             ('val', val_folders),
                             ('test', test_folders)]:
            print(f"Processing {split} data...")
            self._process_split(yolo_dir, split, folders)

        # Create data.yaml with all splits
        self._create_data_yaml(yolo_dir)

    def _process_split(self, yolo_dir: str, split: str, folders: List[str]) -> None:
        """Process a single dataset split"""
        for folder in folders:
            folder_path = os.path.join(self.config.BASE_DIR, folder)
            
            for i in range(8):  # Assuming 8 parts per folder
                # Process image
                img_src = os.path.join(folder_path, f"Part{i}.jpg")
                img_dst = os.path.join(yolo_dir, 'images', split, f"{folder}_Part{i}.jpg")
                shutil.copy2(img_src, img_dst)
                
                # Process mask
                mask_src = os.path.join(folder_path, f"Part{i}_label.bmp")
                label_dst = os.path.join(yolo_dir, 'labels', split, f"{folder}_Part{i}.txt")
                self.mask_to_yolo_label(mask_src, label_dst)

    def _create_data_yaml(self, yolo_dir: str) -> None:
        """Create YOLO dataset configuration file"""
        yaml_content = {
            'path': yolo_dir,
            'train': os.path.join( 'images', 'train'),
            'val': os.path.join( 'images', 'val'),
            'test': os.path.join( 'images', 'test'),
            'names': {0: 'crack'},
            'nc': 1
        }
        
        yaml_path = os.path.join(yolo_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"Created dataset config at {yaml_path}")

class YOLOTrainer:
    def __init__(self, config: Config):
        self.config = config

        self.writer = SummaryWriter(log_dir=os.path.join(
            self.config.LOG_DIR, 
            datetime.now().strftime("%Y%m%d_%H%M%S")))
    def train_model(self, data_yaml: str) -> YOLO:
        """Train YOLO model with comprehensive settings"""
        try:
            # Load model
            model = YOLO('yolov8n-seg.pt')
            
            # Training with augmentation and advanced settings
            results = model.train(
                data=data_yaml,
                epochs=self.config.EPOCHS,
                batch=self.config.BATCH_SIZE,
                imgsz=self.config.IMG_SIZE,
                device=self.config.DEVICE,
                single_cls=True,
                augment=True,
                **self.config.AUGMENTATION,
                optimizer='AdamW',
                lr0=0.001,
                weight_decay=0.0005,
                dropout=0.1,
                patience=10,
                save_period=5,
                resume=False,
                box=7.5,  # box loss gain
                cls=1.0,  # cls loss gain
                dfl=1.5,  # dfl loss gain
                name='yolo_crack_model',
                project=self.config.SAVE_DIR
            )
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    def log_metrics(self, model: YOLO) -> None:
        """Log training metrics and sample predictions"""
        try:
            # Load results CSV
            results_csv = os.path.join(
                self.config.SAVE_DIR, 
                'yolo_crack_model', 
                'results.csv')
            
            if not os.path.exists(results_csv):
                raise FileNotFoundError("Results file not found")
                
            df = pd.read_csv(results_csv)
            
            # Log scalar metrics
            for column in df.columns:
                if any(x in column for x in ['loss', 'precision', 'recall', 'mAP']):
                    for epoch, value in enumerate(df[column].dropna()):
                        self.writer.add_scalar(column, value, epoch)
            
            # Log sample predictions (commented as it requires validation dataloader)
            # self._log_sample_predictions(model)
            
            self.writer.flush()
            
        except Exception as e:
            print(f"Warning: Metric logging failed - {e}")

    def evaluate_model(self, model: YOLO, data_yaml: str) -> Dict:
        """Evaluate model on test set"""
        metrics = model.val(
            data=data_yaml,
            split='test',
            conf=0.25,
            iou=0.6,
            device=self.config.DEVICE
        )
        return {
            'map50': metrics.box.map50,
            'map': metrics.box.map,
            'precision': metrics.box.precision,
            'recall': metrics.box.recall
        }

    def export_model(self, model: YOLO) -> None:
        """Export model to various formats"""
        export_path = os.path.join(self.config.SAVE_DIR, 'exported_models')
        os.makedirs(export_path, exist_ok=True)
        
        # Export to ONNX
        model.export(
            format='onnx',
            dynamic=True,
            simplify=True,
            opset=12,
            imgsz=self.config.IMG_SIZE,
            name='yolo_crack_onnx',
            project=export_path
        )
        
        # Export to TensorRT if CUDA available
        if torch.cuda.is_available():
            model.export(
                format='engine',
                device=0,
                imgsz=self.config.IMG_SIZE,
                name='yolo_crack_engine',
                project=export_path
            )

# @hydra.main(config_path="conf", config_name="config")
def main(): # cfg: AppConfig
    try:
        print("Initializing steel crack detection training pipeline...")
        set_seed(Config.SEED)
        
        # Initialize components
        preparer = YOLODatasetPreparer(Config)
        trainer = YOLOTrainer(Config)
        
        # Prepare dataset
        print("Preparing YOLO dataset structure...")
        yolo_dir = preparer.create_yolo_structure()
        preparer.prepare_dataset(yolo_dir)
        
        # Train model
        print("Starting model training...")
        data_yaml = os.path.join(yolo_dir, 'data.yaml')
        model = trainer.train_model(data_yaml)
        
        # Log results
        print("Logging training metrics...")
        trainer.log_metrics(model)
        
        # Evaluate
        print("Evaluating model...")
        metrics = trainer.evaluate_model(model, data_yaml)
        print(f"Test Metrics: {metrics}")
        
        # Export
        print("Exporting model...")
        trainer.export_model(model)
        
        # Save final model
        model_path = os.path.join(Config.SAVE_DIR, 'best_yolo_crack_model.pt')
        model.save(model_path)
        print(f"Training complete. Model saved to {model_path}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()