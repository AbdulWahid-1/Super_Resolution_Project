# data_prep.py
# This file takes perfect images and ruins them on purpose.
# I add blur, noise, and bad JPEG compression. 
# My AI needs to learn how to fix these exact problems.

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# Simulates saving an image at low quality, like sending it on WhatsApp.
def add_bad_jpeg_compression(image):
    quality = random.randint(30, 80) # Pick a random bad quality
    _, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, 1)

# Simulates a shaky camera or an out-of-focus lens.
def apply_camera_blur(image):
    # 50% chance to do a focus blur
    if random.random() < 0.5:
        kernel = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=random.uniform(0.5, 2.5))
    # 50% chance to do a motion blur (shaky hands)
    else:
        size = random.choice([5, 7, 9])
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size) / size
        return cv2.filter2D(image, -1, kernel)

# This is the main loop where I corrupt my dataset.
def ruin_my_images(hr_folder, lr_folder, scale=4):
    Path(lr_folder).mkdir(parents=True, exist_ok=True)
    images = list(Path(hr_folder).glob("*.png"))
    
    print(f"[+] Found {len(images)} perfect images. Time to mess them up...")
    
    for path in tqdm(images, desc="Corrupting"):
        img = cv2.imread(str(path))
        if img is None: continue
        
        # 1. Crop the image so the math divides perfectly by 4
        h, w, _ = img.shape
        h, w = (h // scale) * scale, (w // scale) * scale
        clean_image = img[:h, :w]
        
        # I overwrite the original to keep sizes perfectly matched.
        cv2.imwrite(str(path), clean_image)
        
        # 2. Add shaky camera blur
        bad_image = apply_camera_blur(clean_image)
        
        # 3. Shrink it down (simulates a cheap, low-res camera)
        bad_image = cv2.resize(bad_image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        
        # 4. Add static noise from the camera sensor
        noise = np.random.normal(0, random.uniform(2, 10), bad_image.shape)
        bad_image = np.clip(bad_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # 5. Finally, compress it badly
        bad_image = add_bad_jpeg_compression(bad_image)
        
        # Save the ruined image to my Low-Res folder
        save_path = Path(lr_folder) / path.name
        cv2.imwrite(str(save_path), bad_image)

if __name__ == "__main__":
    ruin_my_images("datasets/div2k/DIV2K_train_HR", "datasets/div2k/DIV2K_train_LR")
    ruin_my_images("datasets/div2k/DIV2K_valid_HR", "datasets/div2k/DIV2K_valid_LR")
    print("[+] All images are corrupted. Ready for training!")