# FractalTokenizer CUDA backend

The CUDA backend accelerates the irregular operations that previously launched hundreds of small PyTorch kernels:

- information-balanced quantile partitioning;
- ROI-compatible adaptive region sampling, including image gradients;
- inverse variable-region tile rendering, including tile gradients.

The extension is built for the active GPU on first use. On the DGX Spark it targets native `sm_121`. Build it explicitly before training or serving:

```bash
cd /home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
source /home/boni/ai/envs/dgx-dl/bin/activate
./scripts/build_cuda_extension.sh
```

Run the component benchmark and full test suite with:

```bash
./scripts/run_cuda_kernel_benchmark.sh
pytest -q
```

## Runtime configuration

The COCO and ImageNet configuration files expose the performance and preprocessing controls:

```yaml
kernel_backend: auto
partition_density_floor_ratio: 0.02
region_sampling_ratio: 2
detokenizer_postprocess_clamp: false
compile_density: true
compile_decoder: true
compile_refiner: true
```

`kernel_backend` accepts:

- `auto`: use CUDA on CUDA tensors and the PyTorch reference on CPU;
- `cuda`: require CUDA and fail immediately if the extension cannot be used;
- `torch`: force the reference implementation for reproducibility and debugging.

The density temperature, coherence floor, blur, partition density floor, and ROI sampling ratio are all preprocessing controls. Output clamping and refinement are postprocessing controls.

The continuous autoencoder also provides named detokenizer profiles:

```python
from fractaltokenizer_v2 import FractalDetokenizerSettings

fast = FractalDetokenizerSettings.from_profile("fast")
quality = FractalDetokenizerSettings.from_profile("quality")
reference = FractalDetokenizerSettings.from_profile("reference")

fast_image, _ = model.decode(tokens, settings=fast)
quality_image, _ = model.decode(tokens, settings=quality)
reference_image, _ = model.decode(tokens, settings=reference)
```

- `fast` uses the compiled local decoder and CUDA compositor without the expensive spatial refiner.
- `quality` uses the compiled local decoder, CUDA compositor, and compiled refiner.
- `reference` forces eager PyTorch decoding for comparison.

## DGX Spark performance

Primitive forward timings for 256×256 RGB, 96 regions, 16×16 sampled tiles, and 32×32 decoded tiles:

| Operation | Batch | PyTorch | CUDA | Speedup |
|---|---:|---:|---:|---:|
| Quantile partition | 1 | 2.368 ms | 0.086 ms | 27.5× |
| Quantile partition | 32 | 2.383 ms | 0.088 ms | 27.0× |
| Region sampling | 1 | 0.0676 ms | 0.0080 ms | 8.5× |
| Region sampling | 32 | 0.374 ms | 0.182 ms | 2.1× |
| Tile rendering | 1 | 0.348 ms | 0.0090 ms | 38.7× |
| Tile rendering | 32 | 10.478 ms | 0.308 ms | 34.1× |

For the ImageNet classifier at 256×256, 128 tokens, width 384, and BF16:

| Path | Batch 32 forward | Batch 32 forward and backward |
|---|---:|---:|
| Original fractal embed | 10.020 ms | 12.839 ms |
| CUDA fractal embed | 3.192 ms | 4.279 ms |
| Fullgraph CUDA fractal embed | 2.165 ms | 3.006 ms |
| Rectangular 128-token embed | 0.824 ms | 2.071 ms |

At full depth-8 classifier level, the optimized fractal model takes 27.52 ms per batch-32 forward/backward step versus 25.57 ms for the equal-token rectangular ViT: approximately 7.6% overhead instead of the original multi-fold slowdown.

The epoch-10 COCO checkpoint retained reconstruction quality after enabling the optimized backend. On 16 held-out images, the PyTorch reference scored 26.9349 dB and the optimized backend scored 26.9336 dB. Mean absolute output difference was 0.00132.

## Numerical and autograd contract

- Partition metadata is non-differentiable because the partition consumes detached density.
- Region-sampling gradients are returned only for the source image.
- Tile-rendering gradients are returned only for the token tiles.
- Geometry is computed in FP32 for both CUDA and PyTorch backends, including under BF16 autocast.
- Render output is FP32. Reduced-precision tile gradients accumulate in FP32 and are cast once to the tile dtype.
- CUDA backward uses atomic additions and is nondeterministic. Deterministic-algorithm mode raises a clear error.
- Double backward and gradients for boxes or assignments are unsupported.
- Build and warm the extension before CUDA graph capture or the first realtime request.

## Remaining costs

The adaptive geometry operations are no longer the main end-to-end bottleneck. The analytic density model, per-region content CNN, coordinate decoder, and full-resolution refiner contain real neural arithmetic. Density is compiled by default and can be disabled for exact replay. The fast detokenizer skips refinement; quality mode preserves the trained reconstruction path.

The BSP tree backend remains CPU/NumPy-bound and is not accelerated by these kernels. Production speed claims apply to the default quantile backend.

Measured artifacts are written to:

- `runs/cuda_kernels/benchmark.json`
- `runs/cuda_kernels/tokenizer_end_to_end.json`
- `runs/cuda_kernels/checkpoint_validation.json`
- `runs/cuda_kernels/compiled_classifier_smoke.json`
