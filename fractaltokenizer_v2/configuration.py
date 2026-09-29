from __future__ import annotations

from pathlib import Path

import yaml


def load_project_configuration(path: str | Path) -> dict:
    config_path = Path(path)
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError("configuration root must be a mapping")
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("unsupported configuration schema version")
    return document


def load_density_profile(path: str | Path, profile_name: str | None = None) -> tuple[str, dict[str, float]]:
    document = load_project_configuration(path)
    selected_name = document["active_density_profile"] if profile_name is None else profile_name
    profiles = document.get("density_profiles", {})
    if selected_name not in profiles:
        raise ValueError(f"unknown density profile: {selected_name}")
    profile = profiles[selected_name]
    parameters = {
        "density_temperature": float(profile["temperature"]),
        "density_coherence_floor": float(profile["coherence_floor"]),
        "density_blur_sigma": float(profile["blur_sigma"]),
    }
    if parameters["density_temperature"] <= 0:
        raise ValueError("density temperature must be positive")
    if not 0.0 <= parameters["density_coherence_floor"] <= 1.0:
        raise ValueError("density coherence floor must be in [0,1]")
    if parameters["density_blur_sigma"] < 0:
        raise ValueError("density blur sigma cannot be negative")
    return selected_name, parameters


def load_imagenet_training_values(path: str | Path, run_profile: str) -> dict:
    document = load_project_configuration(path)
    runs = document.get("runs", {})
    if run_profile not in runs:
        raise ValueError(f"unknown ImageNet run profile: {run_profile}")
    run = dict(runs[run_profile])
    density_profile_name = run.pop("density_profile", document["active_density_profile"])
    _, density_parameters = load_density_profile(path, density_profile_name)
    values = {**document.get("training_defaults", {}), **run, **density_parameters}
    values["variants"] = tuple(values["variants"])
    values["seeds"] = tuple(int(seed) for seed in values["seeds"])
    values["configuration_file"] = str(Path(path))
    values["run_profile"] = run_profile
    values["density_profile"] = density_profile_name
    return values
