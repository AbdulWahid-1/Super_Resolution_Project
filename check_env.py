import sys
import importlib.util
import platform

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"[-] [MISSING] {package_name} is not installed.")
        return False
    else:
        print(f"[+] [INSTALLED] {package_name} is ready.")
        return True

print("=== SYSTEM & HARDWARE AUDIT ===")
print(f"Python Version: {platform.python_version()}")

required_libraries = {
    "torch": "torch",
    "torchvision": "torchvision",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "scikit-image": "skimage",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm"
}

missing_pkgs = []
for pkg, imp in required_libraries.items():
    if not check_package(pkg, imp):
        missing_pkgs.append(pkg)

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[INFO] GPU Detected: {gpu_name}")
        print(f"[INFO] VRAM Capacity: {vram_gb:.2f} GB")
        if vram_gb < 8.0:
            print("[!] Note: VRAM is under 8GB. Tile-based inference and small batch sizes are strictly required.")
    else:
        print("\n[!] WARNING: Running on CPU. PyTorch cannot find your NVIDIA driver.")
except Exception as e:
    print(f"[-] PyTorch CUDA check failed: {e}")

if missing_pkgs:
    print(f"\n[!] Please install missing packages: pip install {' '.join(missing_pkgs)}")
else:
    print("\n[+] Environment is fully configured and ready!")