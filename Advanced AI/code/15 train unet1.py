import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# مسیر دیتاست
base_dir = r"D:\research\project\AI steel crack\dataset\KolektorSDD"

# تعریف کلاس Dataset
class CrackDataset(Dataset):
    def __init__(self, folders, transform=None, mask_transform=None):
        self.image_mask_pairs = []
        for folder in folders:
            folder_path = os.path.join(base_dir, folder)
            for i in range(8):  # Part0 تا Part7
                img_name = f"Part{i}.jpg"
                mask_name = f"Part{i}_lable.bmp"
                img_path = os.path.join(folder_path, img_name)
                mask_path = os.path.join(folder_path, mask_name)
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.image_mask_pairs.append((img_path, mask_path))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("L")  # grayscale
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# تبدیلات
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# تقسیم پوشه‌ها
train_folders = [f"kos{i:02d}" for i in range(1, 41)]
test_folders = [f"kos{i:02d}" for i in range(41, 51)]

train_dataset = CrackDataset(train_folders, transform=transform, mask_transform=mask_transform)
test_dataset = CrackDataset(test_folders, transform=transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print("تعداد نمونه‌های آموزشی:", len(train_dataset))
print("تعداد نمونه‌های تست:", len(test_dataset))
