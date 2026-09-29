# ImageNet100 classifier comparison

This experiment compares the current adaptive tokenizer against fixed-patch ViTs without changing the transformer or training recipe between methods.

| Variant | Frontend | Tokens at 256×256 | Trainable parameters |
|---|---|---:|---:|
| `fractal_128` | Analytic density, quantile partition, DCT/residual region encoder, geometry encoder | 128 | 14,565,988 |
| `vit_rect_128` | Fixed 16×32 patch projection | 128 | 14,875,876 |
| `vit_patch16_256` | Conventional fixed 16×16 patch projection | 256 | 14,630,116 |

The first comparison controls token count and transformer attention compute. The conventional ViT control shows what happens when a standard square patch embedding is allowed twice as many tokens. All variants use an 8-layer, 384-dimensional, 6-head pre-norm transformer and the same classifier head.

The adaptive classifier intentionally uses continuous region features, not residual vector quantization. There is no trained VQ codebook yet, and including an untrained discrete bottleneck would test codebook collapse rather than adaptive spatial allocation. The analytic density and hard quantile boundaries also mean this experiment tests the current information-allocation heuristic; it does not claim that the partition has been learned from the classification objective.

## Training recipe

- ImageNet100: 130,000 training images and 5,000 validation images in 100 classes.
- 256×256 RandomResizedCrop, horizontal flip, RandAugment, random erasing.
- Mixup 0.8 and label smoothing 0.1.
- AdamW, weight decay 0.05, five warmup epochs, cosine decay, 100 epochs.
- BF16 autocast, pinned persistent workers, four-batch prefetch, cuDNN autotuning.
- Identical effective batch and learning-rate scaling for every model.
- Epoch-boundary resume with model, optimizer, Python, NumPy, Torch, CUDA, and loader RNG state.

The GB10 batch calibration selected micro-batch 64 with eight-step accumulation, giving effective batch 512. Synthetic forward/backward calibration measured approximately 879 images/s for `fractal_128`, 939 images/s for `vit_rect_128`, and 443 images/s for `vit_patch16_256`. The saved steady-state production-path check using staged images, full augmentations, and forward/backward measured approximately 797, 836, and 496 images/s respectively. Exact calibration, memory, and production-path measurements are stored in `batch_plan.json` and `training_path_benchmark.json`. Every training epoch records its own end-to-end throughput.

## Telegram setup

The training process uses `/home/boni/ai/apps/TelegramBot`. Export the credentials in the shell that starts tmux:

```bash
export TELEGRAM_BOT_TOKEN='...'
export TELEGRAM_CHAT_ID='...'
```

The bot sends dataset readiness, run start, every epoch's losses and top-1/top-5, train-validation gap, speed, peak CUDA memory, ETA, periodic curve images, best result, completion, and exception tracebacks. It alerts on non-finite loss, near-chance collapse, likely underfitting, a large generalization gap, and validation stagnation.

`/state` and `/health` are available while training. Telegram terminal commands are explicitly disabled for these runs. If credentials are absent, training continues and retains all local metrics, checkpoints, TensorBoard events, plots, and state files.

## Launch

```bash
ssh dgx-spark
source ~/ai/envs/dgx-dl/bin/activate
cd ~/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
./scripts/launch_imagenet100_comparison.sh
```

The launcher recalibrates the shared micro-batch and starts one resumable tmux job named `fractal-imagenet100`. The first launch validates and stages ImageNet100 from `/mnt/hdd/ImageNet100` to `/dev/shm/ImageNet100`. The source dataset is never modified. `/dev/shm` is ephemeral, so staging runs again after a reboot.

To inspect the process without Telegram:

```bash
tail -f runs/imagenet100_comparison/console.log
tmux attach -t fractal-imagenet100
```

Edit the variables at the top of `scripts/train_imagenet100_comparison.py`; the scripts deliberately do not use command-line arguments. The default performs one seed. For paper results, change `seeds` to `(42, 43, 44)` after the single-seed run has demonstrated stable optimization.

## Artifacts

Each `<variant>_seed<seed>` directory contains:

- `config.json` and `class_to_idx.json`
- `metrics.jsonl`
- `run_state.json`
- `last.pt` and `best.pt`
- `training_curves.png`
- `tensorboard/`

The root contains `batch_plan.json`, `dataset_inventory.json`, `comparison_summary.json`, and `failure.json` if an exception escapes. Checkpoints and JSON state are written atomically.

## Validation commands

```bash
source ~/ai/envs/dgx-dl/bin/activate
cd ~/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
pytest -q
python scripts/smoke_imagenet100_comparison.py
python scripts/tune_imagenet100_batch.py
python scripts/benchmark_imagenet100_training.py
```

The bounded smoke script trains and validates all three architectures for two batches each, exercises checkpoint and plot generation, and never sends Telegram messages.
