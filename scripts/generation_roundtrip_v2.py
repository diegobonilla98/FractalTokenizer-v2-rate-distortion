from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from fractaltokenizer_v2 import FractalTokenPacket, FractalTokenizerV2, FractalTokenizerV2Config


image_path = Path("example.jpg")
checkpoint_path = Path("checkpoints_v2/fractaltokenizer_v2_vq_epoch_100.pt")
packet_path = Path("example.ftk2")
output_path = Path("fractaltokenizer_v2_packet_roundtrip.png")
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
    packet = model.encode_packet(tensor, target_tokens=target_tokens)
    packet.save(packet_path)
    restored = FractalTokenPacket.load(packet_path)
    reconstruction = model.decode_packet(restored)[0].cpu()

rate = restored.estimated_bits(config.codebook_size)
to_pil_image(reconstruction.clamp(0, 1)).save(output_path)
print(f"tokens={int(restored.mask.sum())}")
print(f"content_bits={rate['content_bits']:.1f}")
print(f"geometry_bits={rate['geometry_bits']:.1f}")
print(f"estimated_bpp={rate['bits_per_pixel']:.5f}")
print(packet_path)
print(output_path)
