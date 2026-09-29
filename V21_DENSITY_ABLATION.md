# V2.1 density-only ablation

This experiment keeps the successful v2 tokenizer architecture fixed and changes only the parameter-free density signal.

The three compared variants all use 128 connected rectangular quantile regions, the same CUDA partition and region-sampling operators, the same DCT plus residual region encoder, identical geometry encoding, and the same transformer. Each has 14,565,988 trainable parameters.

- `fractal_analytic_128`: original v2 analytic multiscale residual and coherence density.
- `fractal_universal_128`: four-direction network-free universal residual codelength density.
- `fractal_uniform_128`: image-independent density control passed through the same v2 region encoder and partitioner.

The default run trains the two missing cells, `fractal_universal_128` and `fractal_uniform_128`. The completed analytic v2 and rectangular runs are reused as exact historical controls. The available `v21_density_ablation` profile can rerun all three adaptive variants if replication becomes necessary.

The training recipe reproduces the original v2 ImageNet100 setup: seed 42, 100 epochs, effective batch 512, AdamW, five warmup epochs, Mixup 0.8, label smoothing 0.1, RandAugment, random erasing 0.25, BF16, and no early stopping before epoch 100.

The unchanged historical controls are:

- `runs/imagenet100_comparison/fractal_128_seed42`: 75.02% best top-1.
- `runs/imagenet100_comparison/vit_rect_128_seed42`: 72.18% best top-1.
- `runs/imagenet100_comparison/vit_patch16_256_seed42`: 76.32% best top-1.

The legacy v2 checkpoint loads strictly into `fractal_analytic_128`; eight ImageNet100 validation images produced exactly identical logits and token counts.

The 500-image preflight report is written to `/mnt/hdd/fractaltokenizer_v21/density_audit/audit.json`. It requires exact CUDA and torch partition parity, exactly 128 valid tokens, identical learned architecture, image-independent uniform partitions, image-adaptive analytic and universal partitions, and improved universal-information balance from the universal signal.

Launch with:

```bash
cd /home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
bash scripts/run_imagenet100_v21_density_ablation.sh
```

The launcher rebuilds the CUDA extension, runs the complete test suite, repeats the statistical and visual audit, and then starts resumable training in `v21-density-imagenet100`. It saves only `best.pt` and `last.pt` per variant and reports metrics, plots, alerts, failures, and completion through Telegram.

This is a fixed-rate density ablation. It does not claim variable-rate compression or aggressive merging of all low-information background into a single token.
