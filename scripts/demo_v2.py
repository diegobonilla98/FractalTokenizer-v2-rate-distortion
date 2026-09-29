from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from fractaltokenizer_v2 import FractalTokenizerV2, FractalTokenizerV2Config


image_path = Path("example.jpg")
checkpoint_path = Path("checkpoints_v2/fractaltokenizer_v2_vq_epoch_100.pt")
output_path = Path("fractaltokenizer_v2_demo.png")
image_size = 256

if not image_path.exists():
    raise FileNotFoundError(image_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
tensor = to_tensor(image).unsqueeze(0).to(device)

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FractalTokenizerV2Config(**checkpoint["config"])
    model = FractalTokenizerV2(config)
    model.load_state_dict(checkpoint["model"])
else:
    config = FractalTokenizerV2Config(
        latent_dim=96,
        target_tokens=128,
        token_grid=(8, 16),
        use_vector_quantization=False,
        decoder_feature_dim=64,
        decoder_hidden_dim=96,
        decoder_residual_blocks=2,
    )
    model = FractalTokenizerV2(config)

model = model.to(device).eval()
with torch.inference_mode():
    output = model(tensor)

original = tensor[0].permute(1, 2, 0).cpu().numpy()
basis = output.basis_reconstruction[0].permute(1, 2, 0).cpu().numpy()
neural = output.reconstruction[0].permute(1, 2, 0).cpu().numpy()
density = output.density[0, 0].cpu().numpy()
boxes = output.tokens.boxes[0, output.tokens.mask[0]].cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(original)
for x0, y0, x1, y1 in boxes:
    axes[0].add_patch(
        plt.Rectangle(
            (x0 * image_size, y0 * image_size),
            (x1 - x0) * image_size,
            (y1 - y0) * image_size,
            fill=False,
            linewidth=0.5,
        )
    )
axes[0].set_title(f"Adaptive partition, {len(boxes)} tokens")
axes[1].imshow(density)
axes[1].set_title("Information density")
axes[2].imshow(np.clip(basis, 0, 1))
axes[2].set_title("Deterministic local-basis decode")
axes[3].imshow(np.clip(neural, 0, 1))
axes[3].set_title("Neural Gaussian-splat decode")
for axis in axes:
    axis.axis("off")
fig.tight_layout()
fig.savefig(output_path, dpi=180)
print(output_path)
