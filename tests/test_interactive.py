import numpy as np
import torch

from fractaltokenizer_v2.interactive import (
    controlled_density,
    labels_from_mask,
    merge_adjacent_regions,
    watershed_refine,
)


def test_controlled_density_temperature_is_finite_and_normalized():
    density = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    image = torch.rand(1, 3, 16, 16)
    sharpened = controlled_density(density, image, 0.5, 0.0, 0.0, None, 0.0)
    softened = controlled_density(density, image, 2.0, 0.0, 0.0, None, 0.0)
    assert torch.isfinite(sharpened).all()
    assert torch.allclose(sharpened.mean(), torch.tensor(1.0), atol=1e-6)
    assert sharpened.std() > softened.std()


def test_mask_color_roundtrip_preserves_regions():
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[:, 8:] = (17, 90, 200)
    labels = labels_from_mask(mask, 16)
    assert labels.shape == (16, 16)
    assert len(np.unique(labels)) == 2


def test_watershed_and_merge_keep_valid_label_map():
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[:, 16:] = 1
    guide = np.zeros((32, 32, 3), dtype=np.uint8)
    guide[:, :18] = 64
    guide[:, 18:] = 192
    refined = watershed_refine(labels, guide, radius=4, compactness=0.0)
    merged, count = merge_adjacent_regions(refined, guide, tolerance=0.0)
    assert refined.shape == labels.shape
    assert merged.min() == 0
    assert count == 2
