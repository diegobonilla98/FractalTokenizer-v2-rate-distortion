import json
from pathlib import Path

from fractaltokenizer_v2.configuration import load_imagenet_training_values
from fractaltokenizer_v2.imagenet_training import TrainingConfig, run_comparison


configuration_path = Path("configs/constant_information_imagenet100_overnight.yaml")
run_profile = "overnight"
gate_summary_path = Path("/mnt/hdd/fractaltokenizer_constant_information/universal_codelength_gate/summary.json")


values = load_imagenet_training_values(configuration_path, run_profile)
gate_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
calibrated_mass = float(gate_summary["mass_per_token"]["universal_codelength"])
if abs(float(values["universal_mass_per_token"]) - calibrated_mass) > 1e-6:
    raise RuntimeError("configured universal mass does not match the completed scientific gate")
config = TrainingConfig(**values)
run_comparison(config)
