# FractalTokenizer

FractalTokenizer is research code for adaptive image tokenization. It tests whether image regions should receive tokens according to estimated information instead of equal area. The repository contains an earlier learned adaptive-autoencoder/classification line (v2) and a newer deterministic predictive-information tokenizer (v3). Neither is a validated compression codec or a demonstrated improvement over standard image tokenizers.

## Current status

- **v2:** a broad research scaffold with learned density options, quantile and BSP partitioners, region features, continuous decoders, sequence/prior code, CUDA kernels, and classification experiments. The results below are exploratory and single-seed.
- **v3:** a zero-parameter, network-free predictive information estimator and rectangular variable-rate partitioner. Its synthetic contracts are tested, but the repository does not contain a completed matched-rate downstream result for this path.
- **Rate-distortion claim:** not established. The v2 learned decoder/allocator path is not demonstrated end-to-end rate-distortion optimization. Discrete region boundaries are integer-valued and detached from reconstruction gradients; some v2 experiments use continuous region features instead of a compact coded payload. Estimated packet bits are not a trained entropy-coded bitstream.

## What the code does

V2 estimates a spatial density using analytic multiscale residual/coherence features, a learned surprisal estimator, a distilled density model, or a uniform control. `QuantileGridPartitioner` creates connected rectangular regions at approximately balanced density mass; `BalancedBSPPartitioner` provides a tree partition alternative. Region features can include sampled pixels, DCT coefficients, residual vector quantization, and geometry. The package also contains continuous and grid autoencoders, classifiers, sequence serialization, autoregressive priors, and optional CUDA operators.

V3 uses a deterministic JPEG-LS predictor over one, two, or four scan directions. A fixed residual codelength proxy forms a nonnegative per-pixel signal, which is pooled into atomic cells. A calibrated scalar capacity maps total estimated information to a token count, then the quantile partitioner balances the cell mass across rectangular regions. This is a predictive heuristic, not a learned probability model or a byte-accurate image codec. The partition code includes a regular-grid special case for saturated images whose density is sufficiently stationary.

## Recorded classification evidence

The saved ImageNet100 run used one seed, 256 px inputs, 100 epochs, a shared 8-layer, 384-dimensional transformer, and 128 tokens for the adaptive and rectangular-128 variants. A conventional 16×16 ViT used 256 tokens. The original run reports:

| Variant | Active tokens | Best validation top-1 | Seeds |
| --- | ---: | ---: | ---: |
| v2 analytic adaptive | 128 | 75.02% | 1 |
| Rectangular 16×32 grid | 128 | 72.18% | 1 |
| Square 16×16 grid | 256 | 76.32% | 1 |

The adaptive model was above the equal-token rectangular control in this run, while the 256-token square-grid model scored higher. A later rerun of the adaptive model with a modified density profile scored 73.52%; it is not a direct paired rerun of all controls. These results are historical, single-seed classifier measurements. They do not establish statistical significance, rate-distortion quality, a v3 result, or superiority over current methods. See [`IMAGENET100_COMPARISON.md`](IMAGENET100_COMPARISON.md) and [`V21_DENSITY_ABLATION.md`](V21_DENSITY_ABLATION.md) for experiment setup and scope.

The v3 network-free ImageNet100 overnight run is described in [`CONSTANT_INFORMATION_OVERNIGHT.md`](CONSTANT_INFORMATION_OVERNIGHT.md), but no completed result from that run is included here. Its proposed comparison remains unverified until the uniform, predictive, and rectangular controls finish under the same evaluation and the actual active and padded token counts are reported.

## Install and run

Use Python 3.11 and a compatible PyTorch installation. On the DGX Spark, the project has been used with the existing `dgx-dl` environment; activate it rather than replacing the shared environment:

```bash
cd /home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
source ~/ai/envs/dgx-dl/bin/activate
python -m pip install -e '.[test]'
```

Run the bounded CPU-friendly v3 tests:

```bash
PYTHONPATH=. pytest -q tests/test_fractaltokenizer_v3.py
```

The v2 CUDA operators are optional for CPU paths. To build them on the DGX, set the architecture supported by that machine and run:

```bash
export TORCH_CUDA_ARCH_LIST=12.1
python scripts/build_cuda_extension.py
```

The repository's scripts use editable variables at the top of each file rather than command-line arguments. Review the script and its output paths before running it. The following starts a long, three-variant ImageNet100 training job in a tmux session and requires the staged ImageNet100 dataset, GPU, and Telegram credentials:

```bash
bash scripts/run_constant_information_imagenet100_overnight.sh
```

This is a multi-hour training run. The command is documented for reproducibility; it was not run during this documentation update. Other `launch_*.sh` and `run_*.sh` scripts may also start expensive jobs. Inspect them before use.

## Data, artifacts, and reproducibility

Training datasets, cached sequences, checkpoints, logs, plots, and run outputs are local artifacts and are excluded from Git. Place the required datasets at the paths configured in the selected script/config. Do not put credentials in config files or commit Telegram tokens. ImageNet100, COCO, and Supervisely datasets and any pretrained weights must be obtained and licensed separately.

The saved runs under `runs/` are useful for local analysis but are not part of the source repository. Results can depend on PyTorch/CUDA versions, GPU kernels, preprocessing, dataset splits, and density configuration. The one-seed ImageNet100 numbers above should be reproduced with multiple seeds and matched preprocessing before making comparative claims.

## Repository map

- `fractaltokenizer_v2/`: v2 density, partitioning, token features, decoders, models, training, serialization, and CUDA bindings.
- `fractaltokenizer_v3/`: deterministic predictive-information density and variable-rate partitioning.
- `scripts/`: audits, demos, benchmarks, smoke checks, and training launchers.
- `configs/`: experiment profiles.
- `tests/`: CPU and CUDA tests for the package.
- `runs/`, `build/`, and `archive/`: local outputs or historical material; not included in the published source repository.

## Research limits

A useful next comparison must use matched preprocessing, repeated seeds, fixed and adaptive controls at measured active rates, and include padded-token cost, geometry overhead, latency, and reconstruction or downstream quality. The current repository does not establish that equalizing a chosen density yields equal semantic information, actual entropy-coded rate, or better rate-distortion performance. Historical experiments that fail or disagree are retained locally as evidence; they are not presented as completed results for the current v3 design.
