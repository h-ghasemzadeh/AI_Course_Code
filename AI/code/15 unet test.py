import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Standardized dimensions (must match training)
IMG_HEIGHT = 1248  # Height in pixels
IMG_WIDTH = 480    # Width in pixels

# Output directory
OUTPUT_DIR = r"C:/lecture/AI/support/code/Unet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model path
model_path = os.path.join(OUTPUT_DIR, "unet_epoch_5.pth")

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.down1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = conv_block(128, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv2 = conv_block(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upconv3 = conv_block(64, 32)
        
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        
        # Bridge
        b = self.bridge(p3)
        
        # Decoder with size adjustment
        u1 = self.up1(b)
        u1 = self._pad_to_match(u1, d3)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.upconv1(u1)
        
        u2 = self.up2(u1)
        u2 = self._pad_to_match(u2, d2)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.upconv2(u2)
        
        u3 = self.up3(u2)
        u3 = self._pad_to_match(u3, d1)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.upconv3(u3)
        
        return torch.sigmoid(self.final(u3))
    
    def _pad_to_match(self, x, target):
        """Adjusts size using reflection padding"""
        diff_height = target.size()[2] - x.size()[2]
        diff_width = target.size()[3] - x.size()[3]
        
        padding = [
            diff_width // 2, 
            diff_width - diff_width // 2,
            diff_height // 2,
            diff_height - diff_height // 2
        ]
        return torch.nn.functional.pad(x, padding, mode='reflect')

def load_model():
    """Load and verify the U-Net model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model = UNet()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully")
        return model
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model: {str(e)}\n"
                         "Please ensure the architecture matches the training program exactly")

def preprocess_image(image_path):
    """Preprocess the input image while maintaining grayscale"""
    try:
        image = Image.open(image_path).convert('L')
        
        # Resize to exact dimensions (width, height) for PIL
        if image.size != (IMG_WIDTH, IMG_HEIGHT):
            print(f"Resizing image from {image.size} to ({IMG_WIDTH}, {IMG_HEIGHT})")
            image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        tensor = transform(image).unsqueeze(0)
        print(f"Input tensor shape: {tensor.shape}")  # Should be [1,1,1248,480]
        return tensor
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")

def save_results(input_img, prediction, threshold=0.5):
    """Save all results to files"""
    # Convert tensors to numpy arrays
    input_np = input_img[0, 0].numpy()
    pred_np = prediction[0, 0].numpy()
    
    # Create mask using threshold
    mask = (pred_np > threshold).astype(np.uint8)
    
    # Create overlay (red cracks on grayscale image)
    overlay = np.stack([input_np]*3, axis=-1)  # Convert to RGB
    overlay[mask == 1, 0] = 1.0  # Set red channel where cracks are detected
    overlay[mask == 1, 1:] = 0.0  # Zero out green and blue channels
    
    # Create figure
    plt.figure(figsize=(20, 12))
    
    # Input image
    plt.subplot(2, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_np, cmap='gray')
    plt.axis('off')
    
    # Raw prediction
    plt.subplot(2, 2, 2)
    plt.title("Prediction Heatmap")
    plt.imshow(pred_np, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    # Binary mask
    plt.subplot(2, 2, 3)
    plt.title(f"Binary Mask (Threshold={threshold})")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # Overlay
    plt.subplot(2, 2, 4)
    plt.title("Crack Detection Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the complete figure
    result_fig_path = os.path.join(OUTPUT_DIR, "detection_results.png")
    plt.savefig(result_fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save individual components with proper scaling
    Image.fromarray((input_np * 255).astype(np.uint8)).save(os.path.join(OUTPUT_DIR, "input_image.png"))
    plt.imsave(os.path.join(OUTPUT_DIR, "prediction_heatmap.png"), pred_np, cmap='hot')
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(OUTPUT_DIR, "binary_mask.png"))
    Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(OUTPUT_DIR, "crack_overlay.png"))
    
    # Save prediction data
    np.save(os.path.join(OUTPUT_DIR, "prediction_data.npy"), pred_np)
    
    # Save text report
    with open(os.path.join(OUTPUT_DIR, "results_report.txt"), 'w') as f:
        f.write(f"Crack Detection Results\n{'='*30}\n")
        f.write(f"Input image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}\n")
        f.write(f"Detection threshold: {threshold}\n")
        f.write(f"Total crack area: {np.sum(mask)} pixels\n")
        f.write(f"Max detection confidence: {np.max(pred_np):.4f}\n")
    
    print(f"All results saved to: {OUTPUT_DIR}")

def main():
    try:
        # Load model
        model = load_model()
        
        # Process image
        image_path = r"C:/lecture/AI/support/code/steel_crack.jpg"
        input_tensor = preprocess_image(image_path)
        
        # Verify input dimensions
        if input_tensor.shape[2:] != torch.Size([IMG_HEIGHT, IMG_WIDTH]):
            raise ValueError(f"Input must be {IMG_HEIGHT}x{IMG_WIDTH} but got {input_tensor.shape[2:]}")
        
        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            print("Prediction completed successfully")
        
        # Save all results
        save_results(input_tensor, output)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit()

if __name__ == "__main__":
    main()