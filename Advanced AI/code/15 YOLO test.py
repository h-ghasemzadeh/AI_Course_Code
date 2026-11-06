import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Configuration
MODEL_PATH = os.path.normpath("Yolo/yolo_crack_model/weights/best.pt")
TEST_IMAGE = os.path.normpath("steel_crack_v3.jpg")
SAVE_DIR = os.path.normpath("Yolo/results")
CONF_THRESH = 0.5  # Confidence threshold

# Ensure result directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def debug_model(model):
    """Print model debugging information."""
    print("\n--- Model Info ---")
    print(f"Classes: {model.names}")
    print(f"Number of Classes: {len(model.names)}")
    print(f"Model Task: {model.overrides['task']}")

def debug_image(image_path):
    """Print image debugging information."""
    print("\n--- Image Info ---")
    img = Image.open(image_path)
    print(f"Size: {img.size}")
    print(f"Mode: {img.mode}")
    img_array = np.array(img)
    print(f"Shape: {img_array.shape}")
    print(f"Pixel Stats - Min: {img_array.min()}, Max: {img_array.max()}, Mean: {img_array.mean():.1f}")
    plt.imshow(img_array)
    plt.title("Input Image Preview")
    plt.axis('off')
    plt.show()

def visualize_results(image, results, model_names):
    """Visualize detection results with masks and boxes."""
    vis_image = image.copy()

    for result in results:
        # Draw segmentation masks
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 128] = [0, 255, 0]  # Green mask
                vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.3, 0)

        # Draw bounding boxes and labels
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                color = (255, 0, 0) if cls_id == 0 else (0, 0, 255)

                # Draw rectangle
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # Draw class + confidence label
                label = f"{model_names[cls_id]} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis_image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Add width/height info
                cv2.putText(vis_image, f"Width: {width}px, Height: {height}px", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Add position info
                cv2.putText(vis_image, f"X: {x1}-{x2}, Y: {y1}-{y2}", (x1, y2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return vis_image

def run_detection():
    """Main function to run YOLOv8 detection."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    debug_model(model)

    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(f"Image file not found: {TEST_IMAGE}")
    
    debug_image(TEST_IMAGE)

    image = cv2.imread(TEST_IMAGE)
    if image is None:
        raise ValueError("Failed to load image!")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run prediction
    results = model.predict(
        source=image_rgb,
        conf=CONF_THRESH,
        imgsz=640,
        save=True,
        save_txt=True,
        save_conf=True,
        project=SAVE_DIR,
        name='detection_results',
        exist_ok=True,
        augment=True
    )

    # Print raw detection results
    print("\n--- Raw Detection Results ---")
    for i, det in enumerate(results[0].boxes):
        print(f"Detection {i+1}:")
        print(f"  Class: {model.names[int(det.cls)]} (ID: {int(det.cls)})")
        print(f"  Confidence: {det.conf.item():.4f}")
        print(f"  Position: {det.xyxy[0].tolist()}")
        print(f"  Size: {det.xywh[0].tolist()[2:]} (Width x Height)")

    # Visualize and show results
    visualized = visualize_results(image_rgb, results, model.names)

    plt.figure(figsize=(16, 10))
    plt.imshow(visualized)
    plt.axis('off')
    plt.title("Crack Detection Results")
    plt.show()

    # Save final result image
    result_path = os.path.join(SAVE_DIR, 'final_detection_result.jpg')
    result_bgr = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, result_bgr)

    # Save results as text
    txt_path = os.path.join(SAVE_DIR, 'detection_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:  # ensure Unicode support
        for i, det in enumerate(results[0].boxes):
            f.write(f"Detection {i+1}:\n")
            f.write(f"  Class: {model.names[int(det.cls)]}\n")
            f.write(f"  Confidence: {float(det.conf):.4f}\n")
            position = [float(x) for x in det.xyxy[0]]
            f.write(f"  Position: {position}\n")
            size = [float(x) for x in det.xywh[0][2:]]
            f.write(f"  Size: {size} (Width x Height)\n\n")


    print(f"✅ Results saved:\n  Image: {result_path}\n  Text: {txt_path}")

if __name__ == "__main__":
    run_detection()
