# Constant-information ImageNet100 pilot

This experiment asks whether a network-free codelength measure allocates a limited token budget more usefully than equal-area allocation. It does not use BPE, learned merge scores, labels, or a neural density model.

The three seed-42 variants use the same 256 px images, augmentation, optimizer, effective batch size, Transformer width, depth, heads, schedule, label smoothing, and early-stopping rule:

- `hilbert_uniform_96`: 96 equal-area connected Hilbert intervals.
- `hilbert_universal_96`: variable connected Hilbert intervals with one token per calibrated amount of universal predictive codelength.
- `vit_rect_128`: conventional 16 x 32 rectangular patches, 128 tokens.

The two Hilbert variants have identical 14,828,836-parameter architectures. Each 8 x 8 atomic feature contributes its zeroth and first spatial moments to its owner token. Only the partition density changes. The rectangular model has 14,875,876 parameters, a 0.32% difference.

The universal mass threshold is calibrated without labels on 2,048 balanced ImageNet100 training images. A preflight measured 96.000 training-subset tokens and 96.089 tokens over all 5,000 held-out validation images, with no 160-token cap hits.

The primary comparison is best validation top-1 at the measured active-token budget:

1. Universal versus uniform determines whether codelength allocation helps at the same mean token count.
2. Universal versus rectangular-128 determines whether roughly 25% fewer active tokens retain competitive recognition.
3. Active and padded tokens are both reported. Dense padding is not counted as a speedup.

This is an exploratory one-seed gate. A meaningful positive result is universal beating uniform by at least 1 percentage point, or matching rectangular-128 within 1 point while using at most 80% as many active tokens. Any positive result must be repeated with at least two additional seeds before a paper claim.

The production preflight on the GB10 at batch 64 measured 0.072 s per forward/backward for Hilbert-96, 0.163 s for universal variable tokens on an augmented batch, and 0.061 s for rectangular-128. The 45-epoch three-way run is expected to take roughly 7 to 9 hours, depending on data augmentation and padding. Runs stop only after at least 30 epochs and eight epochs without validation improvement.

Artifacts are stored under `/mnt/hdd/fractaltokenizer_constant_information/imagenet100_overnight`. Each variant keeps only `best.pt` and `last.pt`, JSONL metrics, TensorBoard data, an atomic run state, and training plots. Telegram reports epoch metrics, ETA, active/padded tokens, collapse, underfitting, overfitting, and stagnation.

Launch from the project root with:

```bash
bash scripts/run_constant_information_imagenet100_overnight.sh
```

Monitor with:

```bash
tmux attach -t constant-information-imagenet100
tail -f /mnt/hdd/fractaltokenizer_constant_information/imagenet100_overnight/console.log
```
