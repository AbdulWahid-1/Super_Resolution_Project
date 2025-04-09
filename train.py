# This is the gym where my AI works out and learns.
# I train both the Artist and the Critic here to get razor-sharp results.

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler # Updated to modern PyTorch syntax
from pathlib import Path
from tqdm import tqdm
import random

from model import LightweightSRNet, CriticNet, FeatureLoss

# This handles feeding images into my graphics card.
class MyDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale=4, patch_size=64):
        self.hr_paths = sorted(list(Path(hr_dir).glob("*.png")))
        self.lr_paths = sorted(list(Path(lr_dir).glob("*.png")))
        self.scale = scale
        self.patch_size = patch_size # Small 64x64 chunks to save VRAM

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # Load the bad image and the perfect image
        hr_img = cv2.imread(str(self.hr_paths[idx]))
        lr_img = cv2.imread(str(self.lr_paths[idx]))

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        # I randomly crop a small box from the image so my laptop doesn't crash
        h, w, _ = lr_img.shape
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        lr_patch = lr_img[y:y + self.patch_size, x:x + self.patch_size]
        
        # I crop the exact matching box from the big image
        hr_patch = hr_img[y * self.scale:(y + self.patch_size) * self.scale, 
                          x * self.scale:(x + self.patch_size) * self.scale]

        # PyTorch likes colors first, and numbers between 0 and 1.
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0

        return lr_tensor, hr_tensor

def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Setting up training on: {device}")
    
    # I keep the batch size very small (4) because my RTX 4050 only has 6GB of VRAM.
    batch_size = 4
    epochs = 100
    
    # Load my images
    dataset = MyDataset("datasets/div2k/DIV2K_train_HR", "datasets/div2k/DIV2K_train_LR")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create my Artist, Critic, and the Texture Checker
    artist = LightweightSRNet().to(device)
    critic = CriticNet().to(device)
    texture_checker = FeatureLoss().to(device)
    
    # Math tools to calculate how wrong the AI is
    loss_pixel = nn.L1Loss()
    loss_bce = nn.BCEWithLogitsLoss() # For the Critic's yes/no guess
    
    # Optimizers (the engines that actually update the AI's brain)
    opt_artist = optim.Adam(artist.parameters(), lr=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4)
    
    # This helps save memory on my graphics card (Mixed Precision) - Updated syntax
    scaler = GradScaler('cuda')
    Path("weights").mkdir(exist_ok=True)
    
    print("[+] Starting the main loop...")
    for epoch in range(1, epochs + 1):
        artist.train()
        critic.train()
        
        bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for lr_imgs, hr_imgs in bar:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            # --- 1. TRAIN THE CRITIC ---
            opt_critic.zero_grad()
            with autocast('cuda'): # Updated syntax
                # The artist draws a fake image
                fake_imgs = artist(lr_imgs)
                
                # The critic guesses if the real image is real (should be 1)
                real_guess = critic(hr_imgs)
                critic_real_loss = loss_bce(real_guess, torch.ones_like(real_guess))
                
                # The critic guesses if the fake image is real (should be 0)
                fake_guess = critic(fake_imgs.detach())
                critic_fake_loss = loss_bce(fake_guess, torch.zeros_like(fake_guess))
                
                # Combine the scores
                total_critic_loss = (critic_real_loss + critic_fake_loss) / 2
                
            # Update the critic's brain
            scaler.scale(total_critic_loss).backward()
            scaler.step(opt_critic) # <-- FIXED: Using scaler to step
            
            # --- 2. TRAIN THE ARTIST ---
            opt_artist.zero_grad()
            with autocast('cuda'): # Updated syntax
                # The artist tries to trick the critic into saying 1 (Real)
                trick_guess = critic(fake_imgs)
                gan_loss = loss_bce(trick_guess, torch.ones_like(trick_guess))
                
                # The math loss (are the colors correct?)
                pixel_loss = loss_pixel(fake_imgs, hr_imgs)
                
                # The texture loss (does it look like a natural photo?)
                texture_loss = texture_checker(fake_imgs, hr_imgs)
                
                # Combine all three into one master score
                # I give a heavy weight to pixels, and a tiny weight to the GAN to prevent wild hallucinations.
                total_artist_loss = pixel_loss + (0.005 * texture_loss) + (0.001 * gan_loss)
                
            # Update the artist's brain
            scaler.scale(total_artist_loss).backward()
            scaler.step(opt_artist)  # <-- FIXED: Using scaler to step
            scaler.update()          # Safely flush the scaler
            
            bar.set_postfix({"Artist": f"{total_artist_loss.item():.3f}", "Critic": f"{total_critic_loss.item():.3f}"})
            
        # Save my progress so I don't lose my work!
        if epoch % 5 == 0 or epoch == epochs:
            torch.save(artist.state_dict(), f"weights/best.pt")
            print("[+] Saved my model to weights/best.pt")

if __name__ == "__main__":
    run_training()