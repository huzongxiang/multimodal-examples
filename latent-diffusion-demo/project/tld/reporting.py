from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_metrics(metrics_path: str | Path) -> list[dict]:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    if window == 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _series_to_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    if len(x_values) == 0 or len(y_values) == 0:
        return []
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if y_max == y_min:
        y_max = y_min + 1.0
    x_single_value = x_max == x_min

    points = []
    for x_val, y_val in zip(x_values, y_values):
        if x_single_value:
            px = left + width / 2
        else:
            px = left + width * ((float(x_val) - x_min) / (x_max - x_min))
        y_norm = (float(y_val) - y_min) / (y_max - y_min)
        y_norm = min(1.0, max(0.0, y_norm))
        py = top + height - height * y_norm
        points.append((px, py))
    return points


def _tick_values(vmin: float, vmax: float, count: int = 5) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([0.0, 1.0])
    if vmax == vmin:
        return np.array([vmin], dtype=np.float64)
    return np.linspace(vmin, vmax, count)


def _format_tick(value: float, integer: bool = False, decimals: int = 3) -> str:
    if integer:
        return f"{int(round(value)):,}"
    return f"{value:.{decimals}f}"


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_values: np.ndarray,
    series: list[dict],
    x_label: str,
    y_label: str,
    y_min: float,
    y_max: float,
    x_integer: bool = True,
    y_integer: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    panel_bg = "#f7f8fc"
    border = "#d8ddea"
    axis_color = "#4a5568"
    text_color = "#1d2735"
    grid_color = "#dfe4ef"
    draw.rounded_rectangle(box, radius=22, fill=panel_bg, outline=border, width=2)

    title_font = _load_font(28, bold=True)
    label_font = _load_font(18, bold=False)
    tick_font = _load_font(16, bold=False)

    draw.text((x0 + 28, y0 + 20), title, font=title_font, fill=text_color)
    draw.text((x0 + 28, y0 + 60), y_label, font=label_font, fill="#5b6472")

    plot_left = x0 + 86
    plot_top = y0 + 98
    plot_right = x1 - 28
    plot_bottom = y1 - 76
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    x_ticks = _tick_values(float(x_values.min()) if len(x_values) else 0.0, float(x_values.max()) if len(x_values) else 1.0)
    y_ticks = _tick_values(y_min, y_max)

    for tick in y_ticks:
        y_norm = (float(tick) - y_min) / max(y_max - y_min, 1e-8)
        y_norm = min(1.0, max(0.0, y_norm))
        py = plot_top + plot_height - plot_height * y_norm
        draw.line([(plot_left, py), (plot_right, py)], fill=grid_color, width=1)
        tick_text = _format_tick(tick, integer=y_integer, decimals=3 if not y_integer else 0)
        tw, th = _measure_text(draw, tick_text, tick_font)
        draw.text((plot_left - tw - 12, py - th / 2), tick_text, font=tick_font, fill=axis_color)

    for tick in x_ticks:
        if len(x_ticks) == 1:
            px = plot_left + plot_width / 2
        elif len(x_values):
            px = plot_left + plot_width * ((float(tick) - float(x_ticks[0])) / max(float(x_ticks[-1] - x_ticks[0]), 1.0))
        else:
            px = plot_left
        draw.line([(px, plot_top), (px, plot_bottom)], fill=grid_color, width=1)
        tick_text = _format_tick(tick, integer=x_integer, decimals=0 if x_integer else 2)
        tw, th = _measure_text(draw, tick_text, tick_font)
        draw.text((px - tw / 2, plot_bottom + 10), tick_text, font=tick_font, fill=axis_color)

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom), (plot_right, plot_bottom)], fill="#9ca8bc", width=2)
    draw.text((plot_right - 110, plot_bottom + 38), x_label, font=label_font, fill="#5b6472")

    visible_series = [entry for entry in series if len(entry["x"]) and len(entry["y"])]
    legend_x = max(plot_left + 170, plot_right - 180 * max(1, len(visible_series)))
    legend_y = y0 + 26
    for entry in visible_series:
        if not len(entry["x"]) or not len(entry["y"]):
            continue
        points = _series_to_points(
            entry["x"],
            entry["y"],
            plot_left,
            plot_top,
            plot_width,
            plot_height,
            y_min,
            y_max,
        )
        draw.line(points, fill=entry["color"], width=entry.get("width", 3))
        if entry.get("fill_alpha", 0) > 0:
            pass
        draw.line([(legend_x, legend_y + 9), (legend_x + 24, legend_y + 9)], fill=entry["color"], width=4)
        draw.text((legend_x + 34, legend_y), entry["label"], font=tick_font, fill=axis_color)
        legend_x += 180

    for entry in visible_series:
        if not len(entry["x"]) or not len(entry["y"]):
            continue
        points = _series_to_points(
            entry["x"],
            entry["y"],
            plot_left,
            plot_top,
            plot_width,
            plot_height,
            y_min,
            y_max,
        )
        draw.line(points, fill=entry["color"], width=entry.get("width", 3))
        if entry.get("annotate_last", False):
            lx, ly = points[-1]
            draw.ellipse((lx - 5, ly - 5, lx + 5, ly + 5), fill=entry["color"])
            label = entry.get("last_label", f"{entry['y'][-1]:.4f}")
            tw, th = _measure_text(draw, label, tick_font)
            tx = min(plot_right - tw - 10, lx + 12)
            ty = max(plot_top + 4, ly - th - 12)
            draw.rounded_rectangle((tx - 8, ty - 4, tx + tw + 8, ty + th + 4), radius=10, fill="white", outline=entry["color"], width=2)
            draw.text((tx, ty), label, font=tick_font, fill=entry["color"])


def _plot_training_metrics_pillow(
    output_dir: Path,
    step_x: np.ndarray,
    step_loss: np.ndarray,
    step_smooth: np.ndarray,
    step_loss_clip: float,
    epoch_x: np.ndarray,
    epoch_loss: np.ndarray,
    epoch_seconds: np.ndarray,
    epoch_cuda: np.ndarray,
    best_epoch_idx: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    step_y_max = max(step_loss_clip * 1.08, float(step_smooth.max()) * 1.08 if len(step_smooth) else 1.0)
    epoch_loss_min = max(0.0, float(epoch_loss.min()) * 0.95 if len(epoch_loss) else 0.0)
    epoch_loss_max = float(epoch_loss.max()) * 1.08 if len(epoch_loss) else 1.0
    runtime_max = float(epoch_seconds.max()) * 1.12 if len(epoch_seconds) else 1.0
    cuda_max = float(np.nanmax(epoch_cuda)) * 1.12 if np.isfinite(epoch_cuda).any() else 1.0

    dashboard = Image.new("RGB", (1900, 1180), "white")
    draw = ImageDraw.Draw(dashboard)
    title_font = _load_font(40, bold=True)
    body_font = _load_font(18, bold=False)
    draw.text((66, 38), "Training Dashboard", font=title_font, fill="#19202b")
    subtitle = "Loss, runtime, and memory trends from the recorded training metrics"
    draw.text((68, 92), subtitle, font=body_font, fill="#5b6472")

    panels = [
        (60, 150, 920, 610),
        (980, 150, 1840, 610),
        (60, 660, 920, 1120),
        (980, 660, 1840, 1120),
    ]
    _draw_panel(
        draw,
        panels[0],
        "Step Loss",
        step_x,
        [
            {"x": step_x, "y": step_loss, "color": "#9dbbff", "width": 2, "label": "Raw step loss"},
            {
                "x": step_x,
                "y": step_smooth,
                "color": "#1c5fd4",
                "width": 4,
                "label": "Smoothed loss",
            },
        ],
        "Global Step",
        "MSE Loss",
        0.0,
        step_y_max,
        x_integer=True,
        y_integer=False,
    )
    _draw_panel(
        draw,
        panels[1],
        "Epoch Loss",
        epoch_x,
        [
            {
                "x": epoch_x,
                "y": epoch_loss,
                "color": "#e36f47",
                "width": 4,
                "label": "Epoch loss",
            }
        ],
        "Epoch",
        "Loss",
        epoch_loss_min,
        epoch_loss_max,
        x_integer=True,
        y_integer=False,
    )
    _draw_panel(
        draw,
        panels[2],
        "Epoch Runtime",
        epoch_x,
        [
            {
                "x": epoch_x,
                "y": epoch_seconds,
                "color": "#1e9e74",
                "width": 4,
                "label": "Seconds per epoch",
            }
        ],
        "Epoch",
        "Seconds",
        0.0,
        runtime_max,
        x_integer=True,
        y_integer=False,
    )
    _draw_panel(
        draw,
        panels[3],
        "Peak CUDA Memory",
        epoch_x,
        [
            {
                "x": epoch_x,
                "y": epoch_cuda,
                "color": "#7a52cc",
                "width": 4,
                "label": "Peak allocated MB",
            }
        ],
        "Epoch",
        "MB",
        0.0,
        cuda_max,
        x_integer=True,
        y_integer=True,
    )

    dashboard.save(output_dir / "training_dashboard.png")

    loss_only = dashboard.crop((40, 130, 1860, 620))
    loss_only.save(output_dir / "loss_curves.png")
    runtime_only = dashboard.crop((40, 640, 1860, 1130))
    runtime_only.save(output_dir / "runtime_curves.png")


def write_training_summary(
    metrics_path: str | Path,
    output_dir: str | Path,
    best_model_path: str | Path | None = None,
) -> dict:
    records = load_metrics(metrics_path)
    steps = [record for record in records if record.get("kind") == "step"]
    epochs = [record for record in records if record.get("kind") == "epoch"]
    early_stop = next((record for record in records if record.get("kind") == "early_stop"), None)

    summary = {
        "num_step_records": len(steps),
        "num_epoch_records": len(epochs),
        "early_stopped": early_stop is not None,
        "best_model_path": str(best_model_path) if best_model_path is not None else None,
    }

    if steps:
        summary["first_step_loss"] = steps[0]["loss"]
        summary["last_step_loss"] = steps[-1]["loss"]
        if "cuda_max_allocated_mb" in steps[-1]:
            summary["last_step_cuda_max_allocated_mb"] = steps[-1]["cuda_max_allocated_mb"]

    if epochs:
        best_epoch = min(epochs, key=lambda record: record["mean_loss"])
        summary["first_epoch_mean_loss"] = epochs[0]["mean_loss"]
        summary["last_epoch_mean_loss"] = epochs[-1]["mean_loss"]
        summary["best_epoch"] = best_epoch["epoch"]
        summary["best_epoch_mean_loss"] = best_epoch["mean_loss"]
        summary["last_epoch_seconds"] = epochs[-1]["epoch_seconds"]

    if early_stop is not None:
        summary["early_stop_epoch"] = early_stop["epoch"]
        summary["patience"] = early_stop["patience"]

    output_path = Path(output_dir) / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_training_metrics(metrics_path: str | Path, output_dir: str | Path) -> None:
    records = load_metrics(metrics_path)
    steps = [record for record in records if record.get("kind") == "step"]
    epochs = [record for record in records if record.get("kind") == "epoch"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not steps and not epochs:
        return

    if steps:
        step_x = np.array([record["global_step"] for record in steps], dtype=np.float64)
        step_loss = np.array([record["loss"] for record in steps], dtype=np.float64)
        smooth_window = max(25, min(200, len(step_loss) // 50 or 25))
        step_smooth = moving_average(step_loss, smooth_window)
        step_loss_clip = np.percentile(step_loss, 99.3) if len(step_loss) > 10 else step_loss.max()
    else:
        step_x = np.array([])
        step_loss = np.array([])
        step_smooth = np.array([])
        step_loss_clip = 0.0

    if epochs:
        epoch_x = np.array([record["epoch"] for record in epochs], dtype=np.float64)
        epoch_loss = np.array([record["mean_loss"] for record in epochs], dtype=np.float64)
        epoch_seconds = np.array([record["epoch_seconds"] for record in epochs], dtype=np.float64)
        epoch_cuda = np.array([record.get("cuda_max_allocated_mb", np.nan) for record in epochs], dtype=np.float64)
        best_epoch_idx = int(np.argmin(epoch_loss))
    else:
        epoch_x = np.array([])
        epoch_loss = np.array([])
        epoch_seconds = np.array([])
        epoch_cuda = np.array([])
        best_epoch_idx = 0

    _plot_training_metrics_pillow(
        output_dir=output_dir,
        step_x=step_x,
        step_loss=step_loss,
        step_smooth=step_smooth,
        step_loss_clip=step_loss_clip,
        epoch_x=epoch_x,
        epoch_loss=epoch_loss,
        epoch_seconds=epoch_seconds,
        epoch_cuda=epoch_cuda,
        best_epoch_idx=best_epoch_idx,
    )
