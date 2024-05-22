# scripts/check_gpu.py

import torch

def check_gpus():
    if torch.cuda.is_available():
        print("CUDA is available")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available")

if __name__ == "__main__":
    check_gpus()
