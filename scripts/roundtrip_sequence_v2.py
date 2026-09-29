from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from fractaltokenizer_v2 import FractalTokenizerV2, FractalTokenizerV2Config


image_path = Path("example.jpg")
checkpoint_path = Path("checkpoints_v2/fractaltokenizer_v2_vq_epoch_100.pt")
output_path = Path("fractaltokenizer_v2_sequence_roundtrip.png")
image_size = 256
target_tokens = 128

if not image_path.exists():
    raise FileNotFoundError(image_path)
if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = FractalTokenizerV2Config(**checkpoint["config"])
config.partition_backend = "bsp"
config.target_tokens = target_tokens
config.max_tokens = max(config.max_tokens, target_tokens)
config.use_vector_quantization = True
model = FractalTokenizerV2(config)
model.load_state_dict(checkpoint["model"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
tensor = to_tensor(image).unsqueeze(0).to(device)

with torch.inference_mode():
    sequence = model.encode_sequence(tensor, target_tokens=target_tokens)
    reconstruction = model.decode_sequence(sequence)[0].cpu()

to_pil_image(reconstruction.clamp(0, 1)).save(output_path)
print(f"actions={int(sequence.sequence_length[0])}")
print(f"content_tokens={int(sequence.token_count[0])}")
print(f"codes_shape={tuple(sequence.code_indices.shape)}")
print(output_path)
