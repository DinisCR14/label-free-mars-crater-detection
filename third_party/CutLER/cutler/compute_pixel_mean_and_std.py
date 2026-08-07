import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm

def calculate_mean_std_and_check_channels(dataset_path):
    transform = transforms.ToTensor()
    channel_sum = np.zeros(3)
    channel_squared_sum = np.zeros(3)
    num_images = 0
    identical_channels_count = 0

    for img_name in tqdm(os.listdir(dataset_path), desc="Processing images"):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            continue  # skip non-image files

        img_path = os.path.join(dataset_path, img_name)
        try:
            img = Image.open(img_path)
            original_mode = img.mode
            img = img.convert("RGB")
            img_tensor = transform(img)  # shape: (3, H, W)
        except (UnidentifiedImageError, OSError) as e:
            print(f"⚠️ Skipping {img_name}: {e}")
            continue

        # Check if all channels are identical
        ch0, ch1, ch2 = img_tensor[0], img_tensor[1], img_tensor[2]
        if torch.allclose(ch0, ch1) and torch.allclose(ch1, ch2):
            identical_channels_count += 1

        # Accumulate stats
        channel_sum += img_tensor.mean(dim=[1, 2]).numpy()
        channel_squared_sum += (img_tensor ** 2).mean(dim=[1, 2]).numpy()
        num_images += 1

    if num_images == 0:
        raise ValueError("No valid images found in the dataset path.")

    mean = channel_sum / num_images
    std = np.sqrt(channel_squared_sum / num_images - mean ** 2)

    print(f"\n🧪 Processed {num_images} images")
    print(f"🧾 Images with identical channels: {identical_channels_count} / {num_images}")
    return mean, std

# Run
import torch
dataset_path = '/home/isradmin/Desktop/CutLER/CutLER/datasets/crater_dataset/train'
mean, std = calculate_mean_std_and_check_channels(dataset_path)

print(f"\n✅ Mean: {mean}")
print(f"✅ Std: {std}")
