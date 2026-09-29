from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structures import QuantizerOutput


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        cosine_distance: bool = True,
        ema_decay: float | None = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_threshold: float = 0.1,
        dead_code_check_interval: int = 100,
    ):
        super().__init__()
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if commitment_weight < 0:
            raise ValueError("commitment_weight cannot be negative")
        if ema_decay is not None and not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0,1) or None")
        if ema_epsilon <= 0:
            raise ValueError("ema_epsilon must be positive")
        if dead_code_threshold < 0:
            raise ValueError("dead_code_threshold cannot be negative")
        if dead_code_check_interval < 1:
            raise ValueError("dead_code_check_interval must be positive")
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment_weight = float(commitment_weight)
        self.cosine_distance = bool(cosine_distance)
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.dead_code_threshold = float(dead_code_threshold)
        self.dead_code_check_interval = int(dead_code_check_interval)
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
        nn.init.normal_(self.codebook.weight, std=self.embedding_dim**-0.5)
        if self.ema_decay is not None:
            self.codebook.weight.requires_grad_(False)
            self.register_buffer("ema_count", torch.ones(self.codebook_size))
            self.register_buffer("ema_weight", self.codebook.weight.detach().clone())
        else:
            self.register_buffer("ema_count", torch.empty(0), persistent=False)
            self.register_buffer("ema_weight", torch.empty(0), persistent=False)

    def lookup(self, indices: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if indices.dtype != torch.long:
            indices = indices.long()
        if bool((indices < 0).any()) or bool((indices >= self.codebook_size).any()):
            raise ValueError("code indices are outside the codebook")
        embeddings = self.codebook(indices)
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)
        return embeddings

    def _distributed_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor

    def _replacement_vectors(self, valid: torch.Tensor, count: int) -> torch.Tensor:
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        source = 0
        if distributed:
            world_size = torch.distributed.get_world_size()
            local_count = torch.tensor([valid.shape[0]], device=valid.device, dtype=torch.long)
            gathered = [torch.zeros_like(local_count) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, local_count)
            available = [rank for rank, value in enumerate(gathered) if int(value.item()) > 0]
            if not available:
                return torch.zeros(count, self.embedding_dim, device=valid.device, dtype=self.codebook.weight.dtype)
            source = available[0]
        rank = torch.distributed.get_rank() if distributed else 0
        if rank == source and valid.shape[0] > 0:
            permutation = torch.randperm(valid.shape[0], device=valid.device)
            indices = permutation[torch.arange(count, device=valid.device) % valid.shape[0]]
            replacement = valid.detach()[indices].to(self.codebook.weight.dtype)
        else:
            replacement = torch.zeros(
                count,
                self.embedding_dim,
                device=valid.device,
                dtype=self.codebook.weight.dtype,
            )
        if distributed:
            torch.distributed.broadcast(replacement, src=source)
        return replacement

    @torch.no_grad()
    def _initialize_codebook(self, valid: torch.Tensor) -> None:
        replacement = self._replacement_vectors(valid, self.codebook_size)
        self.codebook.weight.copy_(replacement)
        self.ema_count.fill_(1.0)
        self.ema_weight.copy_(replacement)

    @torch.no_grad()
    def _replace_dead_codes(self, valid: torch.Tensor) -> None:
        if self.dead_code_threshold <= 0:
            return
        if int(self.ema_updates.item()) % self.dead_code_check_interval != 0:
            return
        expired = self.ema_count < self.dead_code_threshold
        count = int(expired.sum().item())
        if count == 0:
            return
        replacement = self._replacement_vectors(valid, count)
        replacement_count = max(self.dead_code_threshold, 1e-3)
        self.ema_count[expired] = replacement_count
        self.ema_weight[expired] = replacement * replacement_count

    @torch.no_grad()
    def _ema_update(self, valid: torch.Tensor, valid_indices: torch.Tensor) -> None:
        update_dtype = self.ema_weight.dtype
        batch_count = torch.bincount(valid_indices, minlength=self.codebook_size).to(update_dtype)
        batch_weight = torch.zeros(
            self.codebook_size,
            self.embedding_dim,
            device=valid.device,
            dtype=update_dtype,
        )
        batch_weight.index_add_(0, valid_indices, valid.detach().to(update_dtype))
        batch_count = self._distributed_sum(batch_count)
        batch_weight = self._distributed_sum(batch_weight)
        decay = self.ema_decay
        self.ema_count.mul_(decay).add_(batch_count, alpha=1.0 - decay)
        self.ema_weight.mul_(decay).add_(batch_weight, alpha=1.0 - decay)
        self.ema_updates.add_(1)
        self._replace_dead_codes(valid)
        total = self.ema_count.sum()
        normalized_count = (
            (self.ema_count + self.ema_epsilon)
            / (total + self.codebook_size * self.ema_epsilon)
            * total
        )
        normalized = self.ema_weight / normalized_count.unsqueeze(-1).clamp_min(self.ema_epsilon)
        self.codebook.weight.copy_(normalized.to(self.codebook.weight.dtype))

    def forward(self, latents: torch.Tensor, mask: torch.Tensor) -> QuantizerOutput:
        if latents.shape[:-1] != mask.shape:
            raise ValueError("latents and mask shapes are incompatible")
        if latents.shape[-1] != self.embedding_dim:
            raise ValueError("latent width does not match embedding_dim")
        flat = latents.reshape(-1, self.embedding_dim)
        flat_mask = mask.reshape(-1).bool()
        valid = flat[flat_mask]
        indices_flat = torch.zeros(flat.shape[0], device=latents.device, dtype=torch.long)
        hard_flat = torch.zeros_like(flat)
        if valid.numel() == 0:
            zero = latents.sum() * 0.0
            return QuantizerOutput(
                quantized=hard_flat.reshape_as(latents),
                indices=indices_flat.reshape(mask.shape),
                loss=zero,
                perplexity=torch.ones((), device=latents.device, dtype=latents.dtype),
            )
        if self.training and torch.is_grad_enabled() and self.ema_decay is not None and int(self.ema_updates.item()) == 0:
            self._initialize_codebook(valid)
        codebook = self.codebook.weight
        if self.cosine_distance:
            distance = 2.0 - 2.0 * F.normalize(valid, dim=-1) @ F.normalize(codebook, dim=-1).t()
        else:
            distance = (
                valid.square().sum(dim=-1, keepdim=True)
                + codebook.square().sum(dim=-1).unsqueeze(0)
                - 2.0 * valid @ codebook.t()
            )
        valid_indices = distance.argmin(dim=-1)
        embedded = self.codebook(valid_indices)
        if self.training and torch.is_grad_enabled() and self.ema_decay is not None:
            self._ema_update(valid, valid_indices)
        if self.ema_decay is None:
            loss = F.mse_loss(embedded, valid.detach())
            loss = loss + self.commitment_weight * F.mse_loss(valid, embedded.detach())
        else:
            loss = self.commitment_weight * F.mse_loss(valid, embedded.detach())
        hard_flat[flat_mask] = embedded
        indices_flat[flat_mask] = valid_indices
        hard = hard_flat.reshape_as(latents)
        straight_through = hard + (latents - latents.detach())
        quantized = torch.where(mask.unsqueeze(-1), straight_through, torch.zeros_like(latents))
        histogram = torch.bincount(valid_indices, minlength=self.codebook_size).to(latents.dtype)
        probability = histogram / histogram.sum().clamp_min(1.0)
        perplexity = torch.exp(-(probability * torch.log(probability + 1e-10)).sum())
        return QuantizerOutput(
            quantized=quantized,
            indices=indices_flat.reshape(mask.shape),
            loss=loss,
            perplexity=perplexity,
        )


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        num_quantizers: int = 1,
        commitment_weight: float = 0.25,
        cosine_distance: bool = True,
        ema_decay: float | None = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_threshold: float = 0.1,
        dead_code_check_interval: int = 100,
    ):
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be positive")
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.num_quantizers = int(num_quantizers)
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=codebook_size,
                    embedding_dim=embedding_dim,
                    commitment_weight=commitment_weight,
                    cosine_distance=cosine_distance,
                    ema_decay=ema_decay,
                    ema_epsilon=ema_epsilon,
                    dead_code_threshold=dead_code_threshold,
                    dead_code_check_interval=dead_code_check_interval,
                )
                for _ in range(num_quantizers)
            ]
        )

    def decode_indices(self, indices: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if indices.ndim == 2 and self.num_quantizers == 1:
            indices = indices.unsqueeze(-1)
        if indices.ndim != 3 or indices.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"indices must have shape [B,N,{self.num_quantizers}], got {tuple(indices.shape)}"
            )
        if mask is not None and mask.shape != indices.shape[:-1]:
            raise ValueError("mask must match the code index prefix")
        embeddings = torch.zeros(
            *indices.shape[:-1],
            self.embedding_dim,
            device=indices.device,
            dtype=self.quantizers[0].codebook.weight.dtype,
        )
        for level, quantizer in enumerate(self.quantizers):
            embeddings = embeddings + quantizer.lookup(indices[..., level])
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)
        return embeddings

    def forward(self, latents: torch.Tensor, mask: torch.Tensor) -> QuantizerOutput:
        residual = latents
        hard_sum = torch.zeros_like(latents)
        indices = []
        losses = []
        perplexities = []
        for quantizer in self.quantizers:
            output = quantizer(residual, mask)
            hard = output.quantized.detach()
            hard_sum = hard_sum + hard
            residual = residual - hard
            indices.append(output.indices)
            losses.append(output.loss)
            perplexities.append(output.perplexity)
        straight_through = hard_sum + (latents - latents.detach())
        quantized = torch.where(mask.unsqueeze(-1), straight_through, torch.zeros_like(latents))
        return QuantizerOutput(
            quantized=quantized,
            indices=torch.stack(indices, dim=-1),
            loss=torch.stack(losses).sum(),
            perplexity=torch.stack(perplexities).mean(),
        )
