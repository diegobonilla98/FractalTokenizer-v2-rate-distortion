# Constant-information contract

## Definition

Given an image \(x\), an information estimator produces \(\rho_x(u) \ge 0\) over atomic cells \(u\). For a partition \(\mathcal P=\{R_1,\ldots,R_N\}\), define

\[
I_i=\sum_{u\in R_i}\rho_x(u), \qquad \bar I=\frac{1}{N}\sum_i I_i.
\]

The primary partition objective is

\[
\mathcal L_{balance}=\frac{1}{N}\sum_i\left(\frac{I_i-\bar I}{\bar I+\epsilon}\right)^2.
\]

For variable-rate tokenization, regions stop growing or splitting around a shared information capacity \(\tau\):

\[
I_i\approx\tau, \qquad N(x)\approx\frac{\sum_u\rho_x(u)}{\tau}.
\]

The information estimator and the partitioner are separate. The partitioner balances mass; it does not merge regions because their appearances are similar.

For the network-free estimator, an atomic cell receives the sum of causal residual code lengths:

\[
\rho_x(u)=\sum_{v\in u}\sum_c-\log_2\hat p_{KT}(r_{v,c}\mid r_{<v,c}).
\]

The active nonrectangular partition follows a known Hilbert path and cuts it at equal cumulative mass. Every interval is connected, and geometry is encoded by its ordered interval boundaries rather than a dense ownership map.

## Non-goals

- Minimizing within-region RGB variance is not the objective.
- Maximizing learned merge frequency is not the objective.
- Matching object boundaries is not sufficient evidence.
- Reducing token count without measuring information balance is not success.
- Reconstruction PSNR alone does not establish constant information.

## Acceptance metrics

Every experiment must report mean token count, information-mass coefficient of variation, maximum-to-mean information mass, fraction of tokens within 20% of the target mass, reconstruction metrics, downstream metrics, geometry rate, tokenizer latency, and total-model latency.

The principal ablation compares information estimators while keeping the same equal-mass partitioner. The principal baseline comparison keeps the same token and compute budget while comparing fixed grid, equal-area adaptive regions, Sobel mass, RGB K-means, connected color clustering, and the proposed information density.
