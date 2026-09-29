from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


study_version = 1
temperature_values = (0.4, 0.55, 0.75, 1.0, 1.35, 1.8, 2.5)
coherence_floor_values = (0.25, 0.35, 0.5, 0.75, 1.0)
blur_sigma_values = (0.0, 0.75, 1.5, 2.5)
parameter_names = ("temperature", "coherence_floor", "blur_sigma")


def configuration_distance(left: dict, right: dict) -> float:
    temperature_distance = abs(math.log(left["temperature"]) - math.log(right["temperature"])) / (
        math.log(max(temperature_values)) - math.log(min(temperature_values))
    )
    coherence_distance = abs(left["coherence_floor"] - right["coherence_floor"]) / (
        max(coherence_floor_values) - min(coherence_floor_values)
    )
    blur_distance = abs(left["blur_sigma"] - right["blur_sigma"]) / max(blur_sigma_values)
    return math.sqrt(temperature_distance**2 + coherence_distance**2 + blur_distance**2)


def build_configurations(count: int = 24, seed: int = 42) -> list[dict]:
    combinations = [
        {
            "temperature": temperature,
            "coherence_floor": coherence_floor,
            "blur_sigma": blur_sigma,
        }
        for temperature in temperature_values
        for coherence_floor in coherence_floor_values
        for blur_sigma in blur_sigma_values
    ]
    default = {"temperature": 1.0, "coherence_floor": 0.35, "blur_sigma": 0.0}
    no_coherence = {"temperature": 1.0, "coherence_floor": 1.0, "blur_sigma": 0.0}
    selected = [default, no_coherence]
    generator = random.Random(seed)
    generator.shuffle(combinations)
    while len(selected) < count:
        remaining = [candidate for candidate in combinations if candidate not in selected]
        scores = [min(configuration_distance(candidate, chosen) for chosen in selected) for candidate in remaining]
        best_score = max(scores)
        best_candidates = [candidate for candidate, score in zip(remaining, scores) if score == best_score]
        selected.append(generator.choice(best_candidates))
    return [{"id": index, **configuration} for index, configuration in enumerate(selected)]


def build_schedule(configurations: list[dict], sample_count: int, seed: int = 42) -> list[dict]:
    generator = random.Random(seed)
    exposure = [0] * len(configurations)
    schedule = []
    for sample_index in range(sample_count):
        minimum_exposure = min(exposure)
        first_pool = [index for index, count in enumerate(exposure) if count == minimum_exposure]
        first = generator.choice(first_pool)
        candidates = [index for index in range(len(configurations)) if index != first]
        best_score = None
        best_candidates = []
        for second in candidates:
            balance = min(exposure) - exposure[second]
            distance = configuration_distance(configurations[first], configurations[second])
            score = 2.0 * balance + distance
            if best_score is None or score > best_score + 1e-12:
                best_score = score
                best_candidates = [second]
            elif abs(score - best_score) <= 1e-12:
                best_candidates.append(second)
        second = generator.choice(best_candidates)
        if generator.random() < 0.5:
            first, second = second, first
        exposure[first] += 1
        exposure[second] += 1
        schedule.append({"sample_index": sample_index, "a": first, "b": second})
    return schedule


def create_state(sample_paths: list[str], token_count: int = 128, seed: int = 42) -> dict:
    configurations = build_configurations(seed=seed)
    return {
        "version": study_version,
        "seed": seed,
        "token_count": token_count,
        "sample_paths": sample_paths,
        "configurations": configurations,
        "schedule": build_schedule(configurations, len(sample_paths), seed=seed),
        "votes": [],
    }


def atomic_json_write(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(values, indent=2))
    temporary_path.replace(path)


def load_or_create_state(path: Path, sample_paths: list[str], token_count: int = 128, seed: int = 42) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("version") == study_version:
            return state
    state = create_state(sample_paths, token_count=token_count, seed=seed)
    atomic_json_write(path, state)
    return state


def record_vote(state: dict, choice: str) -> dict:
    if choice not in {"a", "same", "b"}:
        raise ValueError("choice must be a, same, or b")
    vote_index = len(state["votes"])
    if vote_index >= len(state["schedule"]):
        return state
    trial = state["schedule"][vote_index]
    state["votes"].append(
        {
            "trial": vote_index,
            "sample_index": trial["sample_index"],
            "a": trial["a"],
            "b": trial["b"],
            "choice": choice,
        }
    )
    return state


def undo_vote(state: dict) -> dict:
    if state["votes"]:
        state["votes"].pop()
    return state


def _empty_parameter_statistics() -> dict:
    return {"exposures": 0, "points": 0.0, "wins": 0, "ties": 0, "losses": 0}


def study_statistics(state: dict) -> dict:
    configurations = state["configurations"]
    candidate_statistics = {configuration["id"]: _empty_parameter_statistics() for configuration in configurations}
    parameter_statistics = {
        name: {str(value): _empty_parameter_statistics() for value in values}
        for name, values in [
            ("temperature", temperature_values),
            ("coherence_floor", coherence_floor_values),
            ("blur_sigma", blur_sigma_values),
        ]
    }
    side_choices = {"a": 0, "same": 0, "b": 0}
    for vote in state["votes"]:
        side_choices[vote["choice"]] += 1
        if vote["choice"] == "a":
            scores = {vote["a"]: (1.0, "wins"), vote["b"]: (0.0, "losses")}
        elif vote["choice"] == "b":
            scores = {vote["a"]: (0.0, "losses"), vote["b"]: (1.0, "wins")}
        else:
            scores = {vote["a"]: (0.5, "ties"), vote["b"]: (0.5, "ties")}
        for configuration_id, (points, outcome) in scores.items():
            candidate = candidate_statistics[configuration_id]
            candidate["exposures"] += 1
            candidate["points"] += points
            candidate[outcome] += 1
            configuration = configurations[configuration_id]
            for parameter_name in parameter_names:
                bucket = parameter_statistics[parameter_name][str(configuration[parameter_name])]
                bucket["exposures"] += 1
                bucket["points"] += points
                bucket[outcome] += 1
    for statistics in list(candidate_statistics.values()) + [
        bucket
        for parameter in parameter_statistics.values()
        for bucket in parameter.values()
    ]:
        exposures = statistics["exposures"]
        statistics["preference_score"] = statistics["points"] / exposures if exposures else None
        if exposures:
            score = statistics["preference_score"]
            standard_error = math.sqrt(max(score * (1.0 - score), 0.0) / exposures)
            statistics["ci95_low"] = max(0.0, score - 1.96 * standard_error)
            statistics["ci95_high"] = min(1.0, score + 1.96 * standard_error)
        else:
            statistics["ci95_low"] = None
            statistics["ci95_high"] = None
    ranked_candidates = sorted(
        (
            {**configurations[configuration_id], **statistics}
            for configuration_id, statistics in candidate_statistics.items()
            if statistics["exposures"]
        ),
        key=lambda values: (values["preference_score"], values["exposures"]),
        reverse=True,
    )
    winning_ranges = {}
    for parameter_name, buckets in parameter_statistics.items():
        observed = [
            (float(value), statistics)
            for value, statistics in buckets.items()
            if statistics["exposures"] >= 2 and statistics["preference_score"] is not None
        ]
        if not observed:
            winning_ranges[parameter_name] = None
            continue
        maximum_score = max(statistics["preference_score"] for _, statistics in observed)
        preferred = [value for value, statistics in observed if statistics["preference_score"] >= maximum_score - 0.05]
        winning_ranges[parameter_name] = {
            "minimum": min(preferred),
            "maximum": max(preferred),
            "best_score": maximum_score,
        }
    return {
        "votes": len(state["votes"]),
        "total_trials": len(state["schedule"]),
        "side_choices": side_choices,
        "candidate_statistics": candidate_statistics,
        "parameter_statistics": parameter_statistics,
        "ranked_candidates": ranked_candidates,
        "winning_ranges": winning_ranges,
    }


def report_markdown(state: dict, statistics: dict) -> str:
    lines = [
        "# Fractal tokenization A/B preference report",
        "",
        f"Completed **{statistics['votes']} / {statistics['total_trials']}** blinded comparisons at **{state['token_count']} tokens**.",
        "",
    ]
    if statistics["votes"] == 0:
        lines.append("No votes recorded yet.")
        return "\n".join(lines)
    side = statistics["side_choices"]
    lines.extend(
        [
            f"Position choices: A {side['a']}, same {side['same']}, B {side['b']}.",
            "",
            "## Preferred parameter ranges",
            "",
        ]
    )
    for parameter_name, values in statistics["winning_ranges"].items():
        if values is None:
            lines.append(f"- {parameter_name}: insufficient votes")
        elif values["minimum"] == values["maximum"]:
            lines.append(f"- {parameter_name}: **{values['minimum']:g}**")
        else:
            lines.append(f"- {parameter_name}: **{values['minimum']:g}–{values['maximum']:g}**")
    lines.extend(["", "## Leading joint configurations", "", "| Rank | Temperature | Coherence floor | Blur sigma | Score | Exposure |", "|---:|---:|---:|---:|---:|---:|"])
    for rank, candidate in enumerate(statistics["ranked_candidates"][:8], start=1):
        lines.append(
            f"| {rank} | {candidate['temperature']:g} | {candidate['coherence_floor']:g} | "
            f"{candidate['blur_sigma']:g} | {candidate['preference_score']:.3f} | {candidate['exposures']} |"
        )
    lines.extend(
        [
            "",
            "Scores assign 1 point to a winner and 0.5 to a tie. Ranges include observed values within 0.05 of the best bucket score. "
            "Treat early results as provisional until every bucket has repeated exposure.",
        ]
    )
    return "\n".join(lines)


def report_figure(statistics: dict):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    display_names = {
        "temperature": "Density temperature",
        "coherence_floor": "Coherence floor",
        "blur_sigma": "Blur sigma on 32x32 grid",
    }
    for axis, parameter_name in zip(axes, parameter_names):
        buckets = statistics["parameter_statistics"][parameter_name]
        points = sorted((float(value), values) for value, values in buckets.items())
        x = [value for value, _ in points]
        y = [np.nan if values["preference_score"] is None else values["preference_score"] for _, values in points]
        low = [np.nan if values["ci95_low"] is None else values["ci95_low"] for _, values in points]
        high = [np.nan if values["ci95_high"] is None else values["ci95_high"] for _, values in points]
        axis.plot(x, y, marker="o", color="#4c78a8")
        axis.fill_between(x, low, high, color="#4c78a8", alpha=0.2)
        axis.axhline(0.5, color="black", linewidth=1, linestyle="--")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(display_names[parameter_name])
        axis.set_xlabel("Value")
        axis.set_ylabel("Preference score")
        axis.grid(alpha=0.2)
    return figure


def write_report(state: dict, output_root: Path) -> tuple[str, Path, Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    statistics = study_statistics(state)
    markdown = report_markdown(state, statistics)
    state_path = output_root / "study_state.json"
    report_json_path = output_root / "preference_report.json"
    report_markdown_path = output_root / "preference_report.md"
    votes_path = output_root / "votes.csv"
    plot_path = output_root / "preference_ranges.png"
    atomic_json_write(state_path, state)
    atomic_json_write(report_json_path, statistics)
    report_markdown_path.write_text(markdown)
    with votes_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["trial", "sample_index", "a", "b", "choice"])
        writer.writeheader()
        writer.writerows(state["votes"])
    figure = report_figure(statistics)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return markdown, plot_path, report_markdown_path, report_json_path, votes_path
