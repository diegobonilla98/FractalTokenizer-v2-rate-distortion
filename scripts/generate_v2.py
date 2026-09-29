from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from fractaltokenizer_v2 import (
    FractalAutoregressivePrior,
    FractalPriorConfig,
    FractalTokenizerV2,
    FractalTokenizerV2Config,
)


tokenizer_checkpoint_path = Path("checkpoints_v2/fractaltokenizer_v2_vq_epoch_100.pt")
prior_checkpoint_path = Path("prior_checkpoints_v2/fractal_prior_v2_epoch_100.pt")
output_path = Path("fractaltokenizer_v2_samples.png")
batch_size = 8
image_shape = (256, 256)
grid_shape = None
min_tokens = 64
max_tokens = 192
temperature = 1.0
top_k = 64

if not tokenizer_checkpoint_path.exists():
    raise FileNotFoundError(tokenizer_checkpoint_path)
if not prior_checkpoint_path.exists():
    raise FileNotFoundError(prior_checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_checkpoint = torch.load(tokenizer_checkpoint_path, map_location="cpu", weights_only=False)
tokenizer_config = FractalTokenizerV2Config(**tokenizer_checkpoint["config"])
tokenizer_config.partition_backend = "bsp"
tokenizer_config.use_vector_quantization = True
tokenizer = FractalTokenizerV2(tokenizer_config)
tokenizer.load_state_dict(tokenizer_checkpoint["model"])
tokenizer = tokenizer.to(device).eval()
prior_checkpoint = torch.load(prior_checkpoint_path, map_location="cpu", weights_only=False)
prior = FractalAutoregressivePrior(FractalPriorConfig(**prior_checkpoint["config"]))
prior.load_state_dict(prior_checkpoint["model"])
prior = prior.to(device).eval()
if prior.config.codebook_size != tokenizer_config.codebook_size:
    raise ValueError("prior and tokenizer codebook sizes differ")
if prior.config.num_quantizers != tokenizer_config.num_quantizers:
    raise ValueError("prior and tokenizer quantizer counts differ")
if grid_shape is None:
    grid_shape = (
        (image_shape[0] + tokenizer_config.partition_stride - 1) // tokenizer_config.partition_stride,
        (image_shape[1] + tokenizer_config.partition_stride - 1) // tokenizer_config.partition_stride,
    )

with torch.inference_mode():
    sequence = prior.generate(
        batch_size=batch_size,
        grid_shape=grid_shape,
        image_shape=image_shape,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        sample=True,
        temperature=temperature,
        top_k=top_k,
    )
    reconstruction = tokenizer.decode_sequence(sequence).cpu().clamp(0, 1)

grid = make_grid(reconstruction, nrow=4, padding=2)
to_pil_image(grid).save(output_path)
print(f"tokens={sequence.token_count.tolist()}")
print(output_path)
