import time

import torch

from fractaltokenizer_v2.universal_codelength import HilbertEqualMassPartitioner, UniversalCodelengthDensity


partition_stride = 8
mass_per_token = 9926.331293741861
maximum_tokens = 192
batch_sizes = (1, 8, 16, 32, 64)
repetitions = 3


torch.set_float32_matmul_precision("high")
density_model = UniversalCodelengthDensity(partition_stride).cuda()
partitioner = HilbertEqualMassPartitioner(maximum_tokens).cuda()
print("torch", torch.__version__, torch.cuda.get_device_name())
for batch_size in batch_sizes:
    torch.cuda.reset_peak_memory_stats()
    images = torch.rand(batch_size, 3, 256, 256, device="cuda")
    try:
        density = density_model(images)
        partition = partitioner(density, mass_per_token=mass_per_token)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            density = density_model(images)
        torch.cuda.synchronize()
        density_seconds = (time.perf_counter() - start) / repetitions
        start = time.perf_counter()
        for _ in range(repetitions):
            partition = partitioner(density, mass_per_token=mass_per_token)
        torch.cuda.synchronize()
        partition_seconds = (time.perf_counter() - start) / repetitions
        print(
            batch_size,
            "density_ms",
            density_seconds * 1000.0,
            "density_ms_per_image",
            density_seconds * 1000.0 / batch_size,
            "partition_ms",
            partition_seconds * 1000.0,
            "tokens",
            float(partition.token_count.float().mean()),
            "peak_gib",
            torch.cuda.max_memory_allocated() / 2**30,
        )
    except Exception as error:
        print(batch_size, type(error).__name__, str(error))
    del images
    torch.cuda.empty_cache()
