import os
from PIL import Image
from torchvision import transforms
import torch
import shutil

# تنظیمات
input_base_dir = r"D:/research/project/AI steel crack/dataset/KolektorSDD"
output_base_dir = r"D:/research/project/AI steel crack/dataset/KolektorSDD_resized"
target_size = (480, 1248)  # (width, height)

# ایجاد ساختار دایرکتوری خروجی
os.makedirs(output_base_dir, exist_ok=True)

# تبدیل ابعاد تصاویر و ذخیره آنها
def resize_and_save_images():
    folders = [f"kos{str(i).zfill(2)}" for i in range(1, 51)]  # پوشه‌های kos01 تا kos05
    
    for folder in folders:
        input_folder = os.path.join(input_base_dir, folder)
        output_folder = os.path.join(output_base_dir, folder)
        
        if not os.path.exists(input_folder):
            print(f"پوشه {input_folder} یافت نشد!")
            continue
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"در حال پردازش پوشه {folder}...")
        
        for i in range(8):
            # پردازش تصاویر اصلی
            img_path = os.path.join(input_folder, f"Part{i}.jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("L")
                    img_resized = img.resize(target_size, Image.BILINEAR)
                    img_resized.save(os.path.join(output_folder, f"Part{i}.jpg"))
                except Exception as e:
                    print(f"خطا در پردازش {img_path}: {str(e)}")
            
            # پردازش ماسک‌ها
            mask_path = os.path.join(input_folder, f"Part{i}_label.bmp")
            if os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert("L")
                    mask_resized = mask.resize(target_size, Image.NEAREST)  # از NEAREST برای ماسک‌ها
                    mask_resized.save(os.path.join(output_folder, f"Part{i}_label.bmp"))
                except Exception as e:
                    print(f"خطا در پردازش {mask_path}: {str(e)}")

# اجرای تابع تبدیل ابعاد
print("شروع تغییر ابعاد تصاویر...")
resize_and_save_images()
print("تغییر ابعاد با موفقیت انجام شد!")

