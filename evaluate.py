# This script grades my AI's homework. 
# It compares my AI's fixed image to the perfect original to see how well it did.
# I also updated this to check the file size and image dimensions.

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from pathlib import Path

def grade_my_ai(perfect_path, ai_path, save_dir="output"):
    # First, I get the file sizes in Kilobytes (KB)
    perfect_size_kb = os.path.getsize(perfect_path) / 1024
    ai_size_kb = os.path.getsize(ai_path) / 1024

    # Load both pictures
    perfect_img = cv2.imread(str(perfect_path))
    ai_img = cv2.imread(str(ai_path))
    
    if perfect_img is None or ai_img is None:
        print("[-] Error: Could not find one of the images to compare.")
        return

    # Get the exact pixel dimensions (Width x Height)
    perf_h, perf_w = perfect_img.shape[:2]
    ai_h, ai_w = ai_img.shape[:2]

    # If the sizes are slightly off, I force the AI image to match the perfect one for the math test
    if perfect_img.shape != ai_img.shape:
        ai_img_for_math = cv2.resize(ai_img, (perfect_img.shape[1], perfect_img.shape[0]))
    else:
        ai_img_for_math = ai_img
        
    # Calculate the scores
    # PSNR = Checks basic math colors. SSIM = Checks structure and edges.
    score_psnr = psnr(perfect_img, ai_img_for_math)
    score_ssim = ssim(perfect_img, ai_img_for_math, channel_axis=2)
    
    print(f"=== MY AI'S REPORT CARD ===")
    print(f"Original Size: {perfect_size_kb:.1f} KB | Dimensions: {perf_w}x{perf_h}")
    print(f"Enhanced Size: {ai_size_kb:.1f} KB | Dimensions: {ai_w}x{ai_h}")
    print(f"PSNR Score: {score_psnr:.2f} dB")
    print(f"SSIM Score: {score_ssim:.4f}")
    
    # Now I draw a nice side-by-side graph to put on my GitHub
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left Side: The Original
    axes[0].imshow(cv2.cvtColor(perfect_img, cv2.COLOR_BGR2RGB))
    title_left = f"The Perfect Original\nDimensions: {perf_w} x {perf_h}\nFile Size: {perfect_size_kb:.1f} KB"
    axes[0].set_title(title_left)
    axes[0].axis('off')
    
    # Right Side: The AI Output
    axes[1].imshow(cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB))
    title_right = f"My AI's Work\nDimensions: {ai_w} x {ai_h}\nFile Size: {ai_size_kb:.1f} KB\nPSNR: {score_psnr:.2f} | SSIM: {score_ssim:.4f}"
    axes[1].set_title(title_right)
    axes[1].axis('off')
    
    Path(save_dir).mkdir(exist_ok=True)
    plt.tight_layout()
    out_file = Path(save_dir) / "evaluation_graph.png"
    plt.savefig(out_file, dpi=300)
    print(f"[+] Saved the comparison graph to {out_file}")

if __name__ == "__main__":
    # Point this to a perfect image, and the image my AI tried to fix
    # Change these paths when you are ready to test
    grade_my_ai("test.png", "output/enhanced_test.png")