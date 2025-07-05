import torch
import os

print(f"--- PyTorch Details ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version (PyTorch compiled with): {torch.version.cuda}")
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
print(f"PyTorch installation path: {os.path.dirname(torch.__file__)}")

print(f"\n--- Check for wrap_triton ---")
if hasattr(torch.library, 'wrap_triton'):
    print("SUCCESS: 'torch.library.wrap_triton' IS found in this PyTorch installation.")
else:
    print("FAILURE: 'torch.library.wrap_triton' IS NOT found in this PyTorch installation.")
    print("This indicates a severe problem with the PyTorch installation or environment.")

print(f"\n--- Check sys.path (where Python looks for modules) ---")
import sys
for p in sys.path:
    print(p)