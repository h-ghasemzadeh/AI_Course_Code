import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Standardized image dimensions
IMG_HEIGHT = 1248  # Height
IMG_WIDTH = 480    # Width

class Config:
    BASE_DIR = "dataset/KolektorSDD_resized"
    SAVE_DIR = "Unet"
    LOG_DIR = os.path.join(SAVE_DIR, "../../asr/whisper-runs/Yolo")  # Directory for TensorBoard logs
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)

class SteelCrackDataset(Dataset):
    def __init__(self, base_dir, folder_names):
        self.image_paths = []
        self.mask_paths = []

        for folder in folder_names:
            folder_path = os.path.join(base_dir, folder)
            for i in range(8):
                img_path = os.path.join(folder_path, f"Part{i}.jpg")
                mask_path = os.path.join(folder_path, f"Part{i}_label.bmp")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        mask = mask.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image = transform(image)
        mask = transforms.ToTensor()(mask)
        return image, mask

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

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
        
        # Decoder
        u1 = self.up1(b)
        u1 = self._adjust_size(u1, d3)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.upconv1(u1)
        
        u2 = self.up2(u1)
        u2 = self._adjust_size(u2, d2)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.upconv2(u2)
        
        u3 = self.up3(u2)
        u3 = self._adjust_size(u3, d1)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.upconv3(u3)
        
        return torch.sigmoid(self.final(u3))
    
    def _adjust_size(self, x, target):
        diff_height = target.size()[2] - x.size()[2]
        diff_width = target.size()[3] - x.size()[3]
        
        padding = [
            diff_width // 2, 
            diff_width - diff_width // 2,
            diff_height // 2,
            diff_height - diff_height // 2
        ]
        return torch.nn.functional.pad(x, padding, mode='reflect')

def train_model():
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=Config.LOG_DIR)
    
    train_folders = [f"kos{str(i).zfill(2)}" for i in range(1, 41)]
    test_folders = [f"kos{str(i).zfill(2)}" for i in range(42, 51)]

    train_dataset = SteelCrackDataset(Config.BASE_DIR, train_folders)
    test_dataset = SteelCrackDataset(Config.BASE_DIR, test_folders)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = UNet().to(Config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # Log batch loss
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Log epoch loss and learning rate
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, f"unet_epoch_{epoch+1}.pth"))

    # Close TensorBoard writer
    writer.close()
    
    evaluate_model(model, test_loader)

def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        sample_img, sample_mask = next(iter(test_loader))
        sample_img = sample_img.to(Config.DEVICE)
        pred_mask = model(sample_img).cpu()

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(sample_img[0][0].cpu(), cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(sample_mask[0][0], cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred_mask[0][0], cmap='gray')

        plt.savefig(os.path.join(Config.SAVE_DIR, "sample_result.png"))
        plt.show()

if __name__ == "__main__":
    train_model()