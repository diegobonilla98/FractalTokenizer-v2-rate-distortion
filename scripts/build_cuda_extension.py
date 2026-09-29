import os

import torch

from fractaltokenizer_v2 import build_cuda_extension, cuda_extension_status


CUDA_ARCHITECTURE = "12.1"
VERBOSE = True


os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCHITECTURE
os.environ["FRACTAL_CUDA_VERBOSE"] = "1" if VERBOSE else "0"

module = build_cuda_extension()
torch.cuda.synchronize()
print(module.__file__)
print(cuda_extension_status())
