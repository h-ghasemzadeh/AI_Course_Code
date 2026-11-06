import os
import cv2
import torch
from ultralytics import YOLO
import shutil
from PIL import Image
import numpy as np

# Configuration
class Config:
    BASE_DIR = ""
    SAVE_DIR = ""
    IMG_SIZE = 640
    EPOCHS = 10
    BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_yolo_structure():
    """Create YOLO directory structure with error handling"""
    yolo_dir = os.path.join(Config.SAVE_DIR, 'yolo_data')
    try:
        os.makedirs(os.path.join(yolo_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, 'labels', 'val'), exist_ok=True)
        return yolo_dir
    except Exception as e:
        print(f"Error creating directory structure: {e}")
        raise

def mask_to_yolo_label(mask_path, label_path):
    """Convert mask to YOLO format with validation"""
    try:
        mask = np.array(Image.open(mask_path).convert('L'))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(label_path, 'w') as f:
            for contour in contours:
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                height, width = mask.shape
                points = approx.squeeze()
                points = points / np.array([width, height])
                points = points.flatten().tolist()
                
                if len(points) >= 6:
                    line = '0 ' + ' '.join(map(str, points)) + '\n'
                    f.write(line)
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        raise

def prepare_dataset(yolo_dir):
    """Prepare dataset with file existence checks"""
    train_folders = [f"kos{str(i).zfill(2)}" for i in range(1, 41)]
    val_folders = [f"kos{str(i).zfill(2)}" for i in range(42, 51)]
    
    def process_folder(folders, dataset_type):
        for folder in folders:
            folder_path = os.path.join(Config.BASE_DIR, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: Missing folder {folder_path}")
                continue
                
            for i in range(8):
                # Process image
                img_src = os.path.join(folder_path, f"Part{i}.jpg")
                if not os.path.exists(img_src):
                    print(f"Warning: Missing image {img_src}")
                    continue
                    
                img_dst = os.path.join(yolo_dir, 'images', dataset_type, f"{folder}_Part{i}.jpg")
                try:
                    shutil.copy2(img_src, img_dst)
                except Exception as e:
                    print(f"Error copying {img_src}: {e}")
                    continue
                
                # Process mask
                mask_src = os.path.join(folder_path, f"Part{i}_label.bmp")
                if not os.path.exists(mask_src):
                    print(f"Warning: Missing mask {mask_src}")
                    continue
                    
                label_dst = os.path.join(yolo_dir, 'labels', dataset_type, f"{folder}_Part{i}.txt")
                try:
                    mask_to_yolo_label(mask_src, label_dst)
                except Exception as e:
                    print(f"Error processing mask {mask_src}: {e}")
                    continue
    
    print("Processing training data...")
    process_folder(train_folders, 'train')
    
    print("Processing validation data...")
    process_folder(val_folders, 'val')
    
    # Create data.yaml
    yaml_content = f"""
train: {os.path.join(yolo_dir, 'images', 'train')}
val: {os.path.join(yolo_dir, 'images', 'val')}

names:
  0: crack
    """
    
    yaml_path = os.path.join(yolo_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created dataset config at {yaml_path}")

def train_yolo():
    """Main training function with error handling"""
    try:
        # Prepare dataset
        print("Preparing dataset structure...")
        yolo_dir = prepare_yolo_structure()
        
        print("Preparing dataset files...")
        prepare_dataset(yolo_dir)
        
        # Load model
        print("Loading YOLO model...")
        model = YOLO('yolov8n-seg.pt')  # Segmentation model
        
        # Train
        print("Starting training...")
        results = model.train(
            data=os.path.join(yolo_dir, 'data.yaml'),
            epochs=Config.EPOCHS,
            batch=Config.BATCH_SIZE,
            imgsz=Config.IMG_SIZE,
            project=Config.SAVE_DIR,
            name='yolo_crack_model',
            device=Config.DEVICE,
            single_cls=True
        )
        
        # Save model
        model_path = os.path.join(Config.SAVE_DIR, 'best_yolo_crack_model.pt')
        model.save(model_path)
        print(f"Training complete. Model saved to {model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    print("Starting YOLO training pipeline...")
    train_yolo()
    print("Process completed.")