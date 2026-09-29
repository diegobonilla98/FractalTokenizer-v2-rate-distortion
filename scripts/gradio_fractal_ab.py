from __future__ import annotations

import random
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fractaltokenizer_v2.ab_study import (
    load_or_create_state,
    report_figure,
    report_markdown,
    study_statistics,
    undo_vote,
    write_report,
    record_vote,
)
from fractaltokenizer_v2.density import AnalyticInformationDensity
from fractaltokenizer_v2.interactive import controlled_density, overlay_boundaries, resize_center_crop
from fractaltokenizer_v2.partition import QuantileGridPartitioner


project_root = Path(__file__).resolve().parents[1]
image_root = Path("/mnt/hdd/coco2017/val2017")
output_root = project_root / "runs/fractal_ab_study"
state_path = output_root / "study_state.json"
image_count = 100
image_size = 256
token_count = 128
random_seed = 42
density_floor_ratio = 0.02
server_name = "0.0.0.0"
server_port = 7860
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
density_model = None
study_state = None
image_cache = None
study_lock = threading.Lock()


def selected_paths() -> list[str]:
    paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    generator = random.Random(random_seed)
    generator.shuffle(paths)
    return [str(path) for path in paths[:image_count]]


def load_images(paths: list[str]) -> list[np.ndarray]:
    images = []
    for path in paths:
        with Image.open(path) as image_file:
            image = np.asarray(image_file.convert("RGB"))
        images.append(resize_center_crop(image, image_size))
    return images


def image_tensor(image: np.ndarray) -> torch.Tensor:
    values = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return values.unsqueeze(0).to(device)


def assignment_map(partition) -> np.ndarray:
    assignment = F.interpolate(
        partition.assignment.unsqueeze(1).float(),
        size=(image_size, image_size),
        mode="nearest",
    )
    return assignment[0, 0].long().cpu().numpy()


def display_density_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    left_values = F.interpolate(left, size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
    right_values = F.interpolate(right, size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
    left_values = torch.log(left_values.clamp_min(1e-6)).cpu().numpy()
    right_values = torch.log(right_values.clamp_min(1e-6)).cpu().numpy()
    combined = np.concatenate([left_values.ravel(), right_values.ravel()])
    low, high = np.quantile(combined, [0.02, 0.98])

    def colorize(values: np.ndarray) -> np.ndarray:
        scaled = np.clip((values - low) / max(float(high - low), 1e-8), 0.0, 1.0)
        heat = cv2.applyColorMap((scaled * 255.0).round().astype(np.uint8), cv2.COLORMAP_MAGMA)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    return colorize(left_values), colorize(right_values)


def panel(image: np.ndarray, assignment: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    tokenization = overlay_boundaries(image, assignment, color=(0, 255, 255), thickness=2)
    output = np.concatenate([tokenization, heatmap], axis=1)
    cv2.putText(output, "TOKENS", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "SHARED-SCALE LOG DENSITY", (image_size + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def render_configuration(
    image: torch.Tensor,
    residual_density: torch.Tensor,
    coherence: torch.Tensor,
    variation: torch.Tensor,
    configuration: dict,
):
    density = density_model.compose(
        residual_density,
        coherence,
        variation,
        coherence_floor=configuration["coherence_floor"],
    )
    density = controlled_density(
        density,
        image,
        configuration["temperature"],
        configuration["blur_sigma"],
        0.0,
        None,
        0.0,
    )
    partition = QuantileGridPartitioner(density_floor_ratio).to(device)(density, target_tokens=token_count)
    return density, assignment_map(partition)


def configuration_text(configuration: dict) -> str:
    return (
        f"temperature={configuration['temperature']:g}, "
        f"coherence_floor={configuration['coherence_floor']:g}, "
        f"blur_sigma={configuration['blur_sigma']:g}"
    )


def current_trial(reveal: str = ""):
    trial_index = len(study_state["votes"])
    if trial_index >= len(study_state["schedule"]):
        status = f"Study complete: {len(study_state['votes'])} / {len(study_state['schedule'])} votes recorded."
        if reveal:
            status = reveal + "\n\n" + status
        return None, None, status
    trial = study_state["schedule"][trial_index]
    source = image_cache[trial["sample_index"]]
    configuration_a = study_state["configurations"][trial["a"]]
    configuration_b = study_state["configurations"][trial["b"]]
    image = image_tensor(source)
    with torch.inference_mode():
        residual_density, coherence, variation = density_model.components(image)
        density_a, assignment_a = render_configuration(
            image,
            residual_density,
            coherence,
            variation,
            configuration_a,
        )
        density_b, assignment_b = render_configuration(
            image,
            residual_density,
            coherence,
            variation,
            configuration_b,
        )
    heatmap_a, heatmap_b = display_density_pair(density_a, density_b)
    status = (
        f"**Trial {trial_index + 1} / {len(study_state['schedule'])}** · "
        f"COCO image `{Path(study_state['sample_paths'][trial['sample_index']]).name}` · "
        "parameters are hidden until you vote."
    )
    if reveal:
        status = reveal + "\n\n" + status
    return panel(source, assignment_a, heatmap_a), panel(source, assignment_b, heatmap_b), status


def report_outputs():
    markdown, plot_path, markdown_path, json_path, votes_path = write_report(study_state, output_root)
    return markdown, str(plot_path), [str(markdown_path), str(json_path), str(votes_path)]


def initial_view():
    panel_a, panel_b, status = current_trial()
    markdown, plot_path, files = report_outputs()
    return panel_a, panel_b, status, markdown, plot_path, files


def vote(choice: str):
    with study_lock:
        vote_index = len(study_state["votes"])
        if vote_index >= len(study_state["schedule"]):
            panel_a, panel_b, status = current_trial()
            markdown, plot_path, files = report_outputs()
            return panel_a, panel_b, status, markdown, plot_path, files
        trial = study_state["schedule"][vote_index]
        configuration_a = study_state["configurations"][trial["a"]]
        configuration_b = study_state["configurations"][trial["b"]]
        record_vote(study_state, choice)
        reveal = (
            f"Recorded **{choice.upper()}**. Previous A: `{configuration_text(configuration_a)}`. "
            f"Previous B: `{configuration_text(configuration_b)}`."
        )
        panel_a, panel_b, status = current_trial(reveal)
        markdown, plot_path, files = report_outputs()
        return panel_a, panel_b, status, markdown, plot_path, files


def vote_a():
    return vote("a")


def vote_same():
    return vote("same")


def vote_b():
    return vote("b")


def undo_last_vote():
    with study_lock:
        undo_vote(study_state)
        panel_a, panel_b, status = current_trial("Removed the previous vote.")
        markdown, plot_path, files = report_outputs()
        return panel_a, panel_b, status, markdown, plot_path, files


def refresh_report():
    markdown, plot_path, files = report_outputs()
    return markdown, plot_path, files


def build_app():
    with gr.Blocks(title="FractalTokenizer blinded A/B study") as demo:
        gr.Markdown(
            "# FractalTokenizer blinded A/B preference study\n"
            "Choose the better-looking adaptive partition for the same image. The left half of each panel shows token borders; "
            "the right half shows pure log density using one shared A/B color scale. Token count is fixed at 128."
        )
        status = gr.Markdown()
        with gr.Row(equal_height=True):
            panel_a = gr.Image(label="A · tokenization | density", height=420, buttons=["fullscreen"])
            panel_b = gr.Image(label="B · tokenization | density", height=420, buttons=["fullscreen"])
        with gr.Row():
            button_a = gr.Button("A looks better", variant="primary", scale=2)
            button_same = gr.Button("About the same", variant="secondary", scale=2)
            button_b = gr.Button("B looks better", variant="primary", scale=2)
            undo_button = gr.Button("Undo", variant="secondary", scale=1)
        with gr.Accordion("Live preference report", open=True):
            report = gr.Markdown()
            report_plot = gr.Image(label="Preference score by parameter value")
            report_files = gr.File(label="Download report, JSON, and votes", file_count="multiple")
            refresh_button = gr.Button("Refresh report")
        outputs = [panel_a, panel_b, status, report, report_plot, report_files]
        button_a.click(vote_a, outputs=outputs, concurrency_limit=1)
        button_same.click(vote_same, outputs=outputs, concurrency_limit=1)
        button_b.click(vote_b, outputs=outputs, concurrency_limit=1)
        undo_button.click(undo_last_vote, outputs=outputs, concurrency_limit=1)
        refresh_button.click(refresh_report, outputs=[report, report_plot, report_files], concurrency_limit=1)
        demo.load(initial_view, outputs=outputs)
    return demo


def main():
    global density_model, study_state, image_cache
    paths = selected_paths()
    study_state = load_or_create_state(state_path, paths, token_count=token_count, seed=random_seed)
    image_cache = load_images(study_state["sample_paths"])
    density_model = AnalyticInformationDensity(partition_stride=8).to(device).eval()
    current_trial()
    app = build_app()
    app.queue(default_concurrency_limit=1, max_size=4)
    app.launch(server_name=server_name, server_port=server_port, share=False, show_error=True)


if __name__ == "__main__":
    main()
