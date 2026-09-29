# FractalTokenizer v3: network-free predictive information

V3 contains no learned tokenizer, CNN stem, normalization, vector quantizer, color-symbol grammar, BPE vocabulary, or label-dependent calibration.

For each channel and four rotated scan directions, a deterministic JPEG-LS predictor produces a residual. Image boundaries are predicted from the per-channel image mean, which is treated as one transmitted token statistic. The atomic information contribution is the excess codelength under a fixed Cauchy residual model:

\[
\rho(u)=\sum_{v\in u,c}\frac{1}{4}\sum_d \log_2\left(1+\left(\frac{x_{v,c}-\hat x^{(d)}_{v,c}}{1/255}\right)^2\right).
\]

The subtraction of the zero-residual codelength makes every spatially uniform color exactly zero information. Adding a constant color offset or permuting channels does not alter the signal. A linear ramp pays a small boundary cost rather than receiving free self-predicted boundaries. Spatially i.i.d. noise has uniform expected density.

Given a fixed information capacity \(\tau\), the image rate is

\[
N(x)=\operatorname{clip}\left(\left\lceil\frac{\sum_u\rho_x(u)}{\tau}\right\rceil,1,N_{max}\right).
\]

Rectangular quantile partitioning then balances the measured information across the selected number of tokens. A maximum-rate image uses an exact regular grid only when its atomic information density is spatially stationary, measured by a coefficient of variation no greater than 0.1. A nonstationary image remains adaptive even when it reaches the token cap. The capacity is the only corpus-calibrated scalar and is fitted without labels.

The blocking synthetic contract is:

- Every constant RGB color produces exactly one token.
- I.i.d. Gaussian noise produces the maximum token count and an exact regular grid.
- Increasing unpredictable residual content cannot reduce total measured information.
- Intermediate images produce intermediate rates and information-balanced partitions.
- The tokenizer has zero learned parameters and empty state.

This is a candidate research construction, not a novelty claim. Literature comparison and downstream experiments are required before claiming that it differs materially from all predictive coding, quadtree, entropy partitioning, or adaptive tokenization methods.
