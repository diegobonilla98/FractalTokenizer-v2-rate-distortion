from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structures import FractalOutput


def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return image[..., :, 1:] - image[..., :, :-1], image[..., 1:, :] - image[..., :-1, :]


class FractalRateDistortionLoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        multiscale_weight: float = 0.5,
        gradient_weight: float = 0.25,
        basis_weight: float = 0.2,
        seam_weight: float = 0.1,
        balance_weight: float = 0.05,
        rate_weight: float = 0.01,
        quantizer_weight: float = 1.0,
        charbonnier_epsilon: float = 1e-3,
        codebook_size: int = 8192,
        num_quantizers: int = 1,
    ):
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
        self.multiscale_weight = float(multiscale_weight)
        self.gradient_weight = float(gradient_weight)
        self.basis_weight = float(basis_weight)
        self.seam_weight = float(seam_weight)
        self.balance_weight = float(balance_weight)
        self.rate_weight = float(rate_weight)
        self.quantizer_weight = float(quantizer_weight)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.bits_per_token = math.log2(codebook_size) * num_quantizers

    def _charbonnier(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.charbonnier_epsilon**2).mean()

    def _multiscale(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        current_prediction = prediction
        current_target = target
        for _ in range(3):
            losses.append(F.l1_loss(current_prediction, current_target))
            if min(current_prediction.shape[-2:]) < 4:
                break
            current_prediction = F.avg_pool2d(current_prediction, 2, stride=2)
            current_target = F.avg_pool2d(current_target, 2, stride=2)
        return torch.stack(losses).mean()

    def _gradient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_x, prediction_y = _image_gradients(prediction)
        target_x, target_y = _image_gradients(target)
        return 0.5 * (F.l1_loss(prediction_x, target_x) + F.l1_loss(prediction_y, target_y))

    def _seam(self, prediction: torch.Tensor, target: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        height, width = prediction.shape[-2:]
        owner = F.interpolate(assignment.unsqueeze(1).float(), size=(height, width), mode="nearest")[:, 0].long()
        boundary_x = (owner[:, :, 1:] != owner[:, :, :-1]).unsqueeze(1)
        boundary_y = (owner[:, 1:, :] != owner[:, :-1, :]).unsqueeze(1)
        error_x, error_y = _image_gradients(prediction - target)
        numerator = (error_x.abs() * boundary_x).sum() + (error_y.abs() * boundary_y).sum()
        denominator = boundary_x.sum() * prediction.shape[1] + boundary_y.sum() * prediction.shape[1]
        return numerator / denominator.clamp_min(1)

    def _balance(self, mass: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (mass * mask).sum(dim=1, keepdim=True) / count
        relative = torch.where(mask, mass / (mean + 1e-6), torch.ones_like(mass))
        variance = ((relative - 1.0).square() * mask).sum(dim=1) / count[:, 0]
        return variance.mean()

    def _content_rate(self, output: FractalOutput, image: torch.Tensor) -> torch.Tensor:
        if output.tokens.code_indices is None:
            return output.reconstruction.sum() * 0.0
        token_count = output.tokens.mask.sum(dim=1).to(image.dtype)
        return (token_count * self.bits_per_token / (image.shape[-2] * image.shape[-1])).mean()

    def _geometry_rate(self, output: FractalOutput, image: torch.Tensor) -> torch.Tensor:
        grid_height, grid_width = output.tokens.assignment.shape[-2:]
        coordinate_bits = math.ceil(math.log2(max(grid_height, grid_width) + 1))
        token_count = output.tokens.mask.sum(dim=1).to(image.dtype)
        if output.tokens.trees is not None:
            internal = (token_count - 1.0).clamp_min(0.0)
            topology_bits = 2.0 * token_count - 1.0
            axis_bits = internal
            split_bits = internal * coordinate_bits
            total_bits = topology_bits + axis_bits + split_bits
        else:
            total_bits = token_count * 4.0 * coordinate_bits
        return (total_bits / (image.shape[-2] * image.shape[-1])).mean()

    def forward(self, output: FractalOutput, target: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction = self._charbonnier(output.reconstruction, target)
        multiscale = self._multiscale(output.reconstruction, target)
        gradient = self._gradient(output.reconstruction, target)
        basis = self._charbonnier(output.basis_reconstruction, target)
        if output.tokens.basis_coefficients is None:
            basis_prediction = target.sum() * 0.0
        else:
            mask = output.tokens.mask[:, :, None, None, None]
            difference = F.smooth_l1_loss(
                output.predicted_basis,
                output.tokens.basis_coefficients,
                reduction="none",
            )
            basis_prediction = (difference * mask).sum() / (
                mask.sum().clamp_min(1) * difference.shape[2] * difference.shape[3] * difference.shape[4]
            )
        seam = self._seam(output.reconstruction, target, output.tokens.assignment)
        balance = self._balance(output.tokens.mass, output.tokens.mask)
        content_rate = self._content_rate(output, target)
        geometry_rate = self._geometry_rate(output, target)
        rate = content_rate + geometry_rate
        total = (
            self.reconstruction_weight * reconstruction
            + self.multiscale_weight * multiscale
            + self.gradient_weight * gradient
            + self.basis_weight * (basis + basis_prediction)
            + self.seam_weight * seam
            + self.balance_weight * balance
            + self.rate_weight * rate
            + self.quantizer_weight * output.quantizer_loss
        )
        return {
            "loss": total,
            "reconstruction": reconstruction,
            "multiscale": multiscale,
            "gradient": gradient,
            "basis": basis,
            "basis_prediction": basis_prediction,
            "seam": seam,
            "balance": balance,
            "rate_bpp": rate,
            "content_rate_bpp": content_rate,
            "geometry_rate_bpp": geometry_rate,
            "vq": output.quantizer_loss,
            "perplexity": output.quantizer_perplexity,
            "tokens": output.tokens.token_count.float().mean(),
        }
