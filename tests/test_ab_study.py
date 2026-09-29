from pathlib import Path

import torch

from fractaltokenizer_v2.ab_study import (
    build_configurations,
    build_schedule,
    create_state,
    record_vote,
    study_statistics,
    undo_vote,
    write_report,
)
from fractaltokenizer_v2.density import AnalyticInformationDensity


def test_density_components_reproduce_forward():
    model = AnalyticInformationDensity(partition_stride=8)
    image = torch.rand(2, 3, 64, 64)
    residual, coherence, variation = model.components(image)
    composed = model.compose(residual, coherence, variation)
    assert torch.allclose(model(image), composed)


def test_configurations_and_schedule_are_balanced():
    configurations = build_configurations(count=24, seed=7)
    schedule = build_schedule(configurations, sample_count=100, seed=7)
    assert len({(item["temperature"], item["coherence_floor"], item["blur_sigma"]) for item in configurations}) == 24
    assert all(trial["a"] != trial["b"] for trial in schedule)
    exposure = [0] * len(configurations)
    for trial in schedule:
        exposure[trial["a"]] += 1
        exposure[trial["b"]] += 1
    assert max(exposure) - min(exposure) <= 2


def test_votes_are_side_independent_and_undoable():
    state = create_state(["one.jpg", "two.jpg", "three.jpg"], seed=3)
    first = state["schedule"][0]
    record_vote(state, "a")
    statistics = study_statistics(state)
    assert statistics["candidate_statistics"][first["a"]]["preference_score"] == 1.0
    assert statistics["candidate_statistics"][first["b"]]["preference_score"] == 0.0
    undo_vote(state)
    assert state["votes"] == []


def test_report_artifacts_are_written(tmp_path: Path):
    state = create_state(["one.jpg", "two.jpg"], seed=5)
    record_vote(state, "same")
    markdown, plot_path, markdown_path, json_path, votes_path = write_report(state, tmp_path)
    assert "1 / 2" in markdown
    assert plot_path.exists()
    assert markdown_path.exists()
    assert json_path.exists()
    assert votes_path.exists()
