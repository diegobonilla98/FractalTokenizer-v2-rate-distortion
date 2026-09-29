import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from fractaltokenizer_v2 import (
    FractalAutoregressivePrior,
    FractalPriorConfig,
    FractalSequenceBatch,
    collate_fractal_sequences,
)


sequence_root = Path("sequence_cache_v2")
checkpoint_dir = Path("prior_checkpoints_v2")
metrics_path = checkpoint_dir / "prior_metrics.jsonl"
run_state_path = checkpoint_dir / "run_state.json"
resume_checkpoint_path = None
shards_per_batch = 1
num_workers = 8
epochs = 100
learning_rate = 3e-4
minimum_learning_rate = 3e-5
weight_decay = 0.05
save_every = 1
seed = 42
load_optimizer_state = True

prior_config = FractalPriorConfig(
    codebook_size=8192,
    num_quantizers=1,
    max_grid_size=64,
    model_dim=512,
    depth=12,
    heads=8,
    mlp_ratio=4.0,
    dropout=0.0,
    max_sequence_length=511,
)


class SequenceDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(root.rglob("*.ftseq"))
        if not self.paths:
            raise RuntimeError(f"no .ftseq files found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return FractalSequenceBatch.load(self.paths[index])


torch.manual_seed(seed)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
dataset = SequenceDataset(sequence_root)
example = dataset[0]
if example.code_indices.shape[-1] != prior_config.num_quantizers:
    raise ValueError("cached sequences and prior use different numbers of quantizers")
if int(example.code_indices.max()) >= prior_config.codebook_size:
    raise ValueError("cached sequence contains a code outside the prior vocabulary")
if max(example.grid_shape) > prior_config.max_grid_size:
    raise ValueError("cached sequence grid exceeds max_grid_size")
if int(example.sequence_length.max()) > prior_config.max_sequence_length:
    raise ValueError("cached sequence exceeds max_sequence_length")
loader = DataLoader(
    dataset,
    batch_size=shards_per_batch,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False,
    persistent_workers=num_workers > 0,
    collate_fn=collate_fractal_sequences,
    drop_last=True,
)
model = FractalAutoregressivePrior(prior_config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, epochs),
    eta_min=minimum_learning_rate,
)
start_epoch = 0
if resume_checkpoint_path is not None:
    checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    start_epoch = int(checkpoint.get("epoch", 0))
    if load_optimizer_state and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if load_optimizer_state and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
run_state = {
    "status": "starting",
    "epoch": start_epoch,
    "epochs": epochs,
    "latest_checkpoint": None,
}

for epoch in range(start_epoch, epochs):
    model.train()
    running = {}
    progress = tqdm(loader, desc=f"prior epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
    for step, sequence in enumerate(progress, start=1):
        sequence = sequence.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            losses = model.loss(sequence)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        for name, value in losses.items():
            running[name] = running.get(name, 0.0) + float(value.detach())
        if step % 100 == 0:
            progress.set_postfix({name: f"{value / step:.4f}" for name, value in running.items()})
    scheduler.step()
    epoch_metrics = {
        "epoch": epoch + 1,
        "learning_rate": scheduler.get_last_lr()[0],
        **{name: value / max(len(loader), 1) for name, value in running.items()},
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(epoch_metrics) + "\n")
    if (epoch + 1) % save_every == 0:
        checkpoint_path = checkpoint_dir / f"fractal_prior_v2_epoch_{epoch + 1:03d}.pt"
        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(prior_config),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)
    run_state = {
        "status": "running",
        "epoch": epoch + 1,
        "epochs": epochs,
        "latest_checkpoint": str(checkpoint_path) if (epoch + 1) % save_every == 0 else None,
    }
    temporary_run_state_path = run_state_path.with_suffix(".tmp")
    temporary_run_state_path.write_text(json.dumps(run_state, indent=2), encoding="utf-8")
    temporary_run_state_path.replace(run_state_path)

run_state["status"] = "complete"
temporary_run_state_path = run_state_path.with_suffix(".tmp")
temporary_run_state_path.write_text(json.dumps(run_state, indent=2), encoding="utf-8")
temporary_run_state_path.replace(run_state_path)
