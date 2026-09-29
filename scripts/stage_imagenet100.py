import json

from fractaltokenizer_v2.imagenet_training import TrainingConfig, prepare_imagenet100


source_dataset_root = "/mnt/hdd/ImageNet100"
staged_dataset_root = "/dev/shm/ImageNet100"

config = TrainingConfig(
    source_dataset_root=source_dataset_root,
    staged_dataset_root=staged_dataset_root,
    stage_to_shared_memory=True,
)

data_root, inventory = prepare_imagenet100(config)
print(json.dumps({"data_root": str(data_root), "inventory": inventory}, indent=2, sort_keys=True))
