from importlib import import_module

from .adapter import FractalTokenizerLayerV2
from .baselines import fixed_grid_assignment
from .codec import FractalTokenPacket
from .continuous_autoencoder import (
    ContinuousAutoencoderConfig,
    ContinuousAutoencoderOutput,
    ContinuousFractalAutoencoder,
    ContinuousReconstructionLoss,
    FractalDetokenizerSettings,
)
from .cuda_ops import FractalCudaSettings, build_cuda_extension, cuda_extension_status
from .density import AnalyticInformationDensity, DensityStudent, MaskedSurprisalDensity, UniformInformationDensity
from .generative import (
    FractalSequenceBatch,
    collate_fractal_sequences,
    pack_fractal_sequence,
    trees_to_partition,
    unpack_fractal_sequence,
)
from .grid_autoencoder import GridAutoencoderConfig, GridViTAutoencoder, grid_shape_for_tokens
from .losses import FractalRateDistortionLoss
from .model import FractalTokenizerV2, FractalTokenizerV2Config
from .partition import (
    BalancedBSPPartitioner,
    QuantileGridPartitioner,
    assignment_from_boxes_grid,
    assignment_from_tree,
    deserialize_tree,
    estimate_tree_bits,
    partition_batch_from_boxes,
    partition_batch_from_trees,
    serialize_tree,
)
from .prior import FractalAutoregressivePrior, FractalPriorConfig
from .raster import RasterAutoregressivePrior, RasterPriorConfig, RasterSequenceBatch, collate_raster_sequences
from .structures import FractalOutput, FractalTokens, PartitionBatch, PartitionTree
from .universal_codelength import (
    HilbertEqualMassPartitioner,
    HilbertPartitionBatch,
    UniversalCodelengthDensity,
    hilbert_coordinates,
)

def __getattr__(name: str):
    if name not in {"ClassifierConfig", "count_parameters", "make_classifier"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".classification", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "AnalyticInformationDensity",
    "BalancedBSPPartitioner",
    "ClassifierConfig",
    "ContinuousAutoencoderConfig",
    "ContinuousAutoencoderOutput",
    "ContinuousFractalAutoencoder",
    "ContinuousReconstructionLoss",
    "DensityStudent",
    "FractalAutoregressivePrior",
    "FractalCudaSettings",
    "FractalDetokenizerSettings",
    "FractalOutput",
    "FractalPriorConfig",
    "FractalRateDistortionLoss",
    "FractalSequenceBatch",
    "FractalTokenPacket",
    "FractalTokenizerLayerV2",
    "FractalTokenizerV2",
    "FractalTokenizerV2Config",
    "FractalTokens",
    "GridAutoencoderConfig",
    "GridViTAutoencoder",
    "HilbertEqualMassPartitioner",
    "HilbertPartitionBatch",
    "MaskedSurprisalDensity",
    "PartitionBatch",
    "PartitionTree",
    "QuantileGridPartitioner",
    "RasterAutoregressivePrior",
    "RasterPriorConfig",
    "RasterSequenceBatch",
    "UniformInformationDensity",
    "UniversalCodelengthDensity",
    "assignment_from_boxes_grid",
    "assignment_from_tree",
    "build_cuda_extension",
    "collate_fractal_sequences",
    "collate_raster_sequences",
    "count_parameters",
    "cuda_extension_status",
    "deserialize_tree",
    "estimate_tree_bits",
    "fixed_grid_assignment",
    "grid_shape_for_tokens",
    "hilbert_coordinates",
    "pack_fractal_sequence",
    "make_classifier",
    "partition_batch_from_boxes",
    "partition_batch_from_trees",
    "serialize_tree",
    "trees_to_partition",
    "unpack_fractal_sequence",
]
