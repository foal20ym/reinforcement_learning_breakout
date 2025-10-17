import torch

def print_device_information(torch_device_available):
    if torch_device_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        print(f"🚀 Using GPU: {gpu_name}")
        print(f"  Total Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {gpu_properties.major}.{gpu_properties.minor}")
    else:
        print("⚠️ Using CPU (no GPU detected)")