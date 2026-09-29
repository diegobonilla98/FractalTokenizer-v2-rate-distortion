from __future__ import annotations

import math
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.filters import sobel
from skimage.metrics import structural_similarity
from skimage.segmentation import watershed
from torchvision.transforms.functional import gaussian_blur


def rgb_array(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    if values.shape[2] == 4:
        alpha = values[:, :, 3:4].astype(np.float32) / 255.0
        values = values[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    values = values[:, :, :3]
    if values.dtype != np.uint8:
        maximum = float(values.max()) if values.size else 0.0
        if maximum <= 1.0:
            values = values * 255.0
        values = np.clip(values, 0.0, 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(values)


def resize_center_crop(image: np.ndarray, image_size: int, nearest: bool = False) -> np.ndarray:
    values = np.asarray(image)
    height, width = values.shape[:2]
    scale = image_size / min(height, width)
    resized_height = max(image_size, round(height * scale))
    resized_width = max(image_size, round(width * scale))
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    resized = cv2.resize(values, (resized_width, resized_height), interpolation=interpolation)
    top = (resized_height - image_size) // 2
    left = (resized_width - image_size) // 2
    return np.ascontiguousarray(resized[top : top + image_size, left : left + image_size])


def labels_from_mask(mask: np.ndarray | None, image_size: int) -> np.ndarray | None:
    if mask is None:
        return None
    values = np.asarray(mask)
    values = resize_center_crop(values, image_size, nearest=True)
    if values.ndim == 2:
        _, labels = np.unique(values, return_inverse=True)
    else:
        colors = values[:, :, :3].reshape(-1, 3)
        _, labels = np.unique(colors, axis=0, return_inverse=True)
    return labels.reshape(image_size, image_size).astype(np.int32)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    count = int(labels.max()) + 1
    indices = np.arange(count, dtype=np.uint32)
    colors = np.stack(
        [
            (37 * indices + 17) % 256,
            (73 * indices + 61) % 256,
            (151 * indices + 29) % 256,
        ],
        axis=1,
    ).astype(np.uint8)
    colors[0] = 0
    return colors[labels]


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def overlay_boundaries(
    image: np.ndarray,
    labels: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    output = rgb_array(image).copy()
    boundary = label_boundaries(labels).astype(np.uint8)
    if thickness > 1:
        kernel = np.ones((thickness, thickness), dtype=np.uint8)
        boundary = cv2.dilate(boundary, kernel)
    output[boundary.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return output


def density_heatmap(density: np.ndarray, image: np.ndarray) -> np.ndarray:
    values = density.astype(np.float32)
    low, high = np.quantile(values, [0.02, 0.98])
    scaled = np.clip((values - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    heat = cv2.applyColorMap((scaled * 255.0).round().astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_array(image), 0.42, heat, 0.58, 0.0)


def controlled_density(
    density: torch.Tensor,
    image: torch.Tensor,
    temperature: float,
    blur_sigma: float,
    edge_weight: float,
    segmentation_labels: np.ndarray | None,
    segmentation_edge_weight: float,
) -> torch.Tensor:
    values = density.float().clamp_min(1e-8)
    if blur_sigma > 0.0:
        radius = max(1, math.ceil(3.0 * blur_sigma))
        maximum_radius = max(1, (min(values.shape[-2:]) - 1) // 2)
        radius = min(radius, maximum_radius)
        values = gaussian_blur(values, kernel_size=2 * radius + 1, sigma=blur_sigma)
    values = values / values.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    values = values.clamp_min(1e-6).pow(1.0 / max(float(temperature), 0.05))
    if edge_weight > 0.0:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-12)
        edge = F.interpolate(edge, size=values.shape[-2:], mode="area")
        edge = edge / edge.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        values = values * (1.0 + float(edge_weight) * edge)
    if segmentation_labels is not None and segmentation_edge_weight > 0.0:
        boundary = label_boundaries(segmentation_labels).astype(np.float32)
        boundary = cv2.dilate(boundary, np.ones((5, 5), dtype=np.uint8))
        boundary_tensor = torch.from_numpy(boundary).to(device=values.device, dtype=values.dtype)[None, None]
        boundary_tensor = F.interpolate(boundary_tensor, size=values.shape[-2:], mode="area")
        boundary_tensor = boundary_tensor / boundary_tensor.mean().clamp_min(1e-8)
        values = values * (1.0 + float(segmentation_edge_weight) * boundary_tensor)
    return values / values.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def watershed_refine(
    labels: np.ndarray,
    guide_image: np.ndarray,
    radius: int,
    compactness: float,
) -> np.ndarray:
    if radius <= 0:
        return labels.astype(np.int32, copy=True)
    boundary = label_boundaries(labels).astype(np.uint8)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    band = cv2.dilate(boundary, kernel).astype(bool)
    markers = labels.astype(np.int32) + 1
    markers[band] = 0
    for label in np.unique(labels):
        if np.any(markers == int(label) + 1):
            continue
        coordinates = np.argwhere(labels == label)
        center = coordinates[len(coordinates) // 2]
        markers[center[0], center[1]] = int(label) + 1
    gray = cv2.cvtColor(rgb_array(guide_image), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gradient = sobel(gray)
    refined = watershed(gradient, markers=markers, compactness=float(compactness), watershed_line=False)
    return (refined - 1).astype(np.int32)


def merge_adjacent_regions(
    labels: np.ndarray,
    guide_image: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    compact = inverse.reshape(labels.shape)
    count = len(unique_labels)
    if tolerance <= 0.0 or count <= 1:
        return compact.astype(np.int32), count
    lab = cv2.cvtColor(rgb_array(guide_image).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab = lab / np.asarray([100.0, 255.0, 255.0], dtype=np.float32)
    sums = np.zeros((count, 3), dtype=np.float64)
    sizes = np.bincount(compact.ravel(), minlength=count).astype(np.float64)
    for channel in range(3):
        np.add.at(sums[:, channel], compact.ravel(), lab[:, :, channel].ravel())
    means = sums / np.maximum(sizes[:, None], 1.0)
    pairs_horizontal = np.stack([compact[:, :-1].ravel(), compact[:, 1:].ravel()], axis=1)
    pairs_vertical = np.stack([compact[:-1, :].ravel(), compact[1:, :].ravel()], axis=1)
    pairs = np.concatenate([pairs_horizontal, pairs_vertical], axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    distances = np.linalg.norm(means[pairs[:, 0]] - means[pairs[:, 1]], axis=1)
    parent = np.arange(count, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for pair_index in np.argsort(distances):
        if distances[pair_index] > tolerance:
            break
        left, right = pairs[pair_index]
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    roots = np.asarray([find(index) for index in range(count)])
    _, mapping = np.unique(roots, return_inverse=True)
    merged = mapping[compact]
    return merged.astype(np.int32), int(mapping.max()) + 1


def mean_color_decode(labels: np.ndarray, guide_image: np.ndarray) -> np.ndarray:
    guide = rgb_array(guide_image).astype(np.float32)
    count = int(labels.max()) + 1
    sizes = np.bincount(labels.ravel(), minlength=count).astype(np.float32)
    colors = np.zeros((count, 3), dtype=np.float32)
    for channel in range(3):
        colors[:, channel] = np.bincount(
            labels.ravel(),
            weights=guide[:, :, channel].ravel(),
            minlength=count,
        ) / np.maximum(sizes, 1.0)
    return np.clip(colors[labels], 0.0, 255.0).round().astype(np.uint8)


def pca_token_map(latents: torch.Tensor, assignment: np.ndarray) -> np.ndarray:
    values = latents[0].float()
    values = values - values.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(values, q=3, center=False)
    colors = values @ basis[:, :3]
    low = torch.quantile(colors, 0.02, dim=0, keepdim=True)
    high = torch.quantile(colors, 0.98, dim=0, keepdim=True)
    colors = ((colors - low) / (high - low).clamp_min(1e-8)).clamp(0.0, 1.0)
    colors = (colors.cpu().numpy() * 255.0).round().astype(np.uint8)
    return colors[assignment]


def image_metrics(reference: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    reference_float = rgb_array(reference).astype(np.float32) / 255.0
    decoded_float = rgb_array(decoded).astype(np.float32) / 255.0
    mse = float(np.mean((reference_float - decoded_float) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    ssim = float(structural_similarity(reference_float, decoded_float, channel_axis=2, data_range=1.0))
    return psnr, ssim


def segmentation_metrics(
    predicted: np.ndarray,
    target: np.ndarray | None,
    boundary_tolerance: int = 2,
) -> dict[str, float]:
    if target is None:
        return {}
    predicted_boundary = label_boundaries(predicted)
    target_boundary = label_boundaries(target)
    if predicted_boundary.any() and target_boundary.any():
        target_distance = distance_transform_edt(~target_boundary)
        predicted_distance = distance_transform_edt(~predicted_boundary)
        precision = float((target_distance[predicted_boundary] <= boundary_tolerance).mean())
        recall = float((predicted_distance[target_boundary] <= boundary_tolerance).mean())
        boundary_f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    else:
        precision = 0.0
        recall = 0.0
        boundary_f1 = 0.0
    correct = 0
    for region in np.unique(predicted):
        target_values = target[predicted == region]
        correct += int(np.bincount(target_values).max())
    purity = correct / predicted.size
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": boundary_f1,
        "region_purity": purity,
    }


def error_heatmap(reference: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    difference = np.abs(rgb_array(reference).astype(np.float32) - rgb_array(decoded).astype(np.float32)).mean(axis=2)
    high = max(float(np.quantile(difference, 0.98)), 1.0)
    scaled = np.clip(difference / high, 0.0, 1.0)
    heat = cv2.applyColorMap((scaled * 255.0).round().astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def timed_call(function, *arguments):
    start = perf_counter()
    result = function(*arguments)
    return result, (perf_counter() - start) * 1000.0
