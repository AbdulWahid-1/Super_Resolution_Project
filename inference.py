# I use this script to take one bad image and let my trained Artist fix it.

import torch
import cv2
import numpy as np
from pathlib import Path
from model import LightweightSRNet # I only need the Artist for this step

def upscale_image(model_path, image_path, save_path):
    # Let's see if we have a GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Loading my AI on: {device}")
    
    # Setup my Artist and load the brain weights I just trained
    artist = LightweightSRNet().to(device)
    
    if not Path(model_path).exists():
        print("[-] I can't find the weights file! Did I run train.py first?")
        return
        
    artist.load_state_dict(torch.load(model_path, map_location=device))
    artist.eval() # Tell the AI we are testing, not training
    
    # Load the bad image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[-] Could not find the image at {image_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert it to the math format PyTorch likes
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    
    # Ask the AI to fix it (without tracking gradients to save memory)
    print("[+] Upscaling the image now...")
    with torch.no_grad():
        output = artist(tensor)
        
    # Convert the math back into a regular picture
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Save it!
    Path(save_path).parent.mkdir(exist_ok=True)
    cv2.imwrite(str(save_path), output)
    print(f"[+] Done! Saved the sharp image to: {save_path}")

if __name__ == "__main__":
    # I use Path to make sure Windows doesn't get confused by folders
    BASE_DIR = Path(__file__).resolve().parent
    
    # Change 'test.png' to whatever bad image you want to fix
    bad_image = BASE_DIR / "test.jpg" 
    good_image = BASE_DIR / "output" / "enhanced_test.jpg"
    my_weights = BASE_DIR / "weights" / "best.pt"
    
    upscale_image(my_weights, bad_image, good_image)