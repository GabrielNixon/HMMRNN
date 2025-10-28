"""Generate lightweight SVG visualisations for synthetic pipeline runs.

The script reads the ``history.json`` and ``metrics.json`` files emitted by
``series_hmm_rnn.run_synthetic_pipeline`` and writes a handful of SVG figures
without relying on heavy plotting libraries (useful in network-restricted
environments).  The SVGs are intentionally simple so they can be embedded in
Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

COLOR_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
STATE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
BINARY_LIGHT = "#e6ecf3"
BINARY_DARK = "#3b6fb6"
ACTION_COLORS = {0: "#4C78A8", 1: "#F58518"}
TRANSITION_COLORS = {1: "#54A24B", 0: "#E45756"}


def load_history(path: Path) -> Sequence[Mapping[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics(path: Path) -> Mapping[str, Mapping[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sample_trace(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_posterior_trace(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _linear_ticks(min_value: float, max_value: float, count: int = 5) -> List[float]:
    if math.isclose(min_value, max_value):
        min_value -= 0.5
        max_value += 0.5
    if math.isclose(min_value, round(min_value)) and math.isclose(max_value, round(max_value)):
        min_int = int(round(min_value))
        max_int = int(round(max_value))
        if count <= 1:
            return [float(min_int)]
        step = (max_int - min_int) / (count - 1)
        ticks = [min_int + round(step * i) for i in range(count)]
        ticks[-1] = max_int
        return [float(value) for value in ticks]
    step = (max_value - min_value) / max(1, count - 1)
    return [min_value + i * step for i in range(count)]


def _format_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{int(round(value))}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def line_plot(
    histories: Sequence[Mapping[str, object]],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plot_width = 640
    plot_height = 360
    margin_left, margin_bottom, margin_top, margin_right = 60, 50, 30, 20
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    points = []
    labels: List[str] = []
    for entry in histories:
        history = entry.get("history", [])
        if not isinstance(history, Iterable):
            continue
        filtered = [record for record in history if isinstance(record, Mapping) and metric in record]
        if not filtered:
            continue
        epochs = [record["epoch"] for record in filtered]
        values = [record[metric] for record in filtered]
        labels.append(str(entry.get("label", "Run")))
        points.append((epochs, values))

    if not points:
        return

    x_min = min(min(series[0]) for series in points)
    x_max = max(max(series[0]) for series in points)
    y_min = min(min(series[1]) for series in points)
    y_max = max(max(series[1]) for series in points)

    if math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    if metric.endswith("accuracy"):
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 1.0)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5

    def scale_x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * inner_width

    def scale_y(value: float) -> float:
        return plot_height - margin_bottom - (value - y_min) / (y_max - y_min) * inner_height

    max_epochs = max(len(series[0]) for series in points)
    x_ticks = _linear_ticks(x_min, x_max, count=min(6, max_epochs))
    y_ticks = _linear_ticks(y_min, y_max)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        f'<rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff"/>',
        f'<text x="{plot_width / 2}" y="20" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>',
        f'<text x="{margin_left / 2}" y="{plot_height / 2}" transform="rotate(-90 {margin_left / 2},{plot_height / 2})" text-anchor="middle" font-size="12" font-family="sans-serif">{ylabel}</text>',
        f'<line x1="{margin_left}" y1="{plot_height - margin_bottom}" x2="{plot_width - margin_right}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
    ]

    # Grid and tick labels
    for tick in x_ticks:
        x = scale_x(tick)
        svg_parts.append(
            f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{plot_height - margin_bottom}" stroke="#e0e0e0" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{x}" y="{plot_height - margin_bottom + 20}" text-anchor="middle" font-size="11" font-family="sans-serif">{_format_float(tick)}</text>'
        )

    for tick in y_ticks:
        y = scale_y(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{plot_width - margin_right}" y2="{y}" stroke="#e0e0e0" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-size="11" font-family="sans-serif">{_format_float(tick)}</text>'
        )

    # Plot lines
    for idx, (epochs, values) in enumerate(points):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        coords = " ".join(
            f"{scale_x(float(x)):.2f},{scale_y(float(y)):.2f}" for x, y in zip(epochs, values)
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'
        )
        for x, y in zip(epochs, values):
            svg_parts.append(
                f'<circle cx="{scale_x(float(x)):.2f}" cy="{scale_y(float(y)):.2f}" r="3" fill="{color}"/>'
            )

    # Legend
    legend_x = margin_left + 10
    legend_y = margin_top + 10
    for idx, label in enumerate(labels):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y + idx * 18}" width="12" height="12" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y + idx * 18 + 10}" font-size="11" font-family="sans-serif">{label}</text>'
        )

    svg_parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def bar_chart(
    runs: Sequence[Mapping[str, object]],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    categories: List[str] = []
    values: List[float] = []
    for entry in runs:
        label = str(entry.get("label", "Run"))
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for split in ("train", "test"):
            split_metrics = metrics.get(split, {})
            if isinstance(split_metrics, Mapping) and metric in split_metrics:
                categories.append(f"{label} ({split})")
                values.append(float(split_metrics[metric]))

    if not values:
        return

    plot_width = 640
    plot_height = 360
    margin_left, margin_bottom, margin_top, margin_right = 80, 60, 30, 20
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    y_min = min(values)
    y_max = max(values)
    if metric.endswith("accuracy"):
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 1.0)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5

    y_ticks = _linear_ticks(y_min, y_max)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        f'<rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff"/>',
        f'<text x="{plot_width / 2}" y="20" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>',
        f'<text x="{margin_left / 2}" y="{plot_height / 2}" transform="rotate(-90 {margin_left / 2},{plot_height / 2})" text-anchor="middle" font-size="12" font-family="sans-serif">{ylabel}</text>',
        f'<line x1="{margin_left}" y1="{plot_height - margin_bottom}" x2="{plot_width - margin_right}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
    ]

    def scale_y(value: float) -> float:
        return plot_height - margin_bottom - (value - y_min) / (y_max - y_min) * inner_height

    for tick in y_ticks:
        y = scale_y(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{plot_width - margin_right}" y2="{y}" stroke="#e0e0e0" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-size="11" font-family="sans-serif">{_format_float(tick)}</text>'
        )

    bar_width = inner_width / max(1, len(values)) * 0.6
    spacing = inner_width / max(1, len(values))

    for idx, (category, value) in enumerate(zip(categories, values)):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        x = margin_left + spacing * idx + (spacing - bar_width) / 2
        y = scale_y(value)
        height = plot_height - margin_bottom - y
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{height}" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width / 2}" y="{y - 5}" text-anchor="middle" font-size="11" font-family="sans-serif">{value:.3f}</text>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width / 2}" y="{plot_height - margin_bottom + 15}" text-anchor="middle" font-size="11" font-family="sans-serif">{category}</text>'
        )

    svg_parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def _state_segments(states: Sequence[int]) -> List[tuple]:
    segments: List[tuple] = []
    if not states:
        return segments
    start = 0
    current = states[0]
    for idx in range(1, len(states)):
        value = states[idx]
        if value != current:
            segments.append((start, idx, current))
            start = idx
            current = value
    segments.append((start, len(states), current))
    return segments


def sequence_overview_plot(
    trace: Mapping[str, Sequence[int]],
    metadata: Mapping[str, object],
    out_path: Path,
    *,
    max_steps: int = 80,
) -> None:
    states = trace.get("states")
    actions = trace.get("actions")
    transitions = trace.get("transitions")
    rewards = trace.get("rewards")
    if not isinstance(states, Sequence) or not states:
        return
    steps = min(len(states), max_steps)
    if steps <= 0:
        return

    def _slice(seq):
        if isinstance(seq, Sequence):
            return [int(seq[i]) for i in range(min(len(seq), steps))]
        return []

    states_slice = _slice(states)
    actions_slice = _slice(actions)
    transitions_slice = _slice(transitions)
    rewards_slice = _slice(rewards)

    plot_width = 820
    plot_height = 320
    margin_left, margin_top, margin_right, margin_bottom = 80, 40, 20, 60
    inner_width = plot_width - margin_left - margin_right
    row_height = 40
    col_width = inner_width / steps

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        '<text x="410" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">Synthetic Trace Overview</text>',
    ]

    rows = [
        ("Latent phase", states_slice, lambda v: STATE_COLORS[v % len(STATE_COLORS)]),
        ("Action", actions_slice, lambda v: ACTION_COLORS.get(v, ACTION_COLORS.get(v % 2, "#999"))),
        ("Transition", transitions_slice, lambda v: TRANSITION_COLORS.get(v, "#999")),
        ("Reward", rewards_slice, lambda v: BINARY_DARK if v else BINARY_LIGHT),
    ]

    for row_idx, (label, values, color_fn) in enumerate(rows):
        y_base = margin_top + row_idx * row_height
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y_base + row_height / 2 + 5}" text-anchor="end" font-size="12" '
            f'font-family="sans-serif">{label}</text>'
        )
        for step_idx in range(steps):
            value = values[step_idx] if step_idx < len(values) else 0
            x = margin_left + step_idx * col_width
            color = color_fn(value)
            svg_parts.append(
                f'<rect x="{x:.2f}" y="{y_base}" width="{col_width:.2f}" height="{row_height - 6}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="0.5"/>'
            )

    # time axis ticks
    tick_count = min(10, steps)
    for tick in range(tick_count + 1):
        step = int(round(tick * (steps - 1) / max(1, tick_count)))
        x = margin_left + step * col_width
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top + len(rows) * row_height}" '
            f'x2="{x:.2f}" y2="{margin_top + len(rows) * row_height + 6}" stroke="#000"/>'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{margin_top + len(rows) * row_height + 22}" text-anchor="middle" '
            f'font-size="11" font-family="sans-serif">t={step}</text>'
        )

    metadata_lines = []
    dwell = metadata.get("dwell")
    if dwell is not None:
        metadata_lines.append(f"dwell≈{dwell}")
    p_common = metadata.get("p_common")
    if p_common is not None:
        metadata_lines.append(f"p_common={p_common}")
    beta = metadata.get("beta")
    if beta is not None:
        metadata_lines.append(f"β={beta}")
    sticky = metadata.get("sticky")
    if sticky is not None:
        metadata_lines.append(f"sticky={sticky}")
    summary = ", ".join(metadata_lines)
    if summary:
        svg_parts.append(
            f'<text x="{plot_width / 2}" y="{plot_height - margin_bottom + 35}" text-anchor="middle" '
            f'font-size="12" font-family="sans-serif">{summary}</text>'
        )

    svg_parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def transition_matrix_plot(sticky: float, dwell: Optional[int], out_path: Path) -> None:
    stay = float(sticky)
    switch = 1.0 - stay
    matrix = [[stay, switch], [switch, stay]]
    plot_width = 360
    plot_height = 280
    cell_size = 120
    margin_left = 100
    margin_top = 80

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        '<text x="180" y="30" text-anchor="middle" font-size="16" font-family="sans-serif">Sticky Transition Matrix</text>',
        '<text x="70" y="70" text-anchor="middle" font-size="13" font-family="sans-serif">from →</text>',
        '<text x="180" y="60" text-anchor="middle" font-size="13" font-family="sans-serif">to phase</text>',
    ]

    for row in range(2):
        y = margin_top + row * cell_size
        svg_parts.append(
            f'<text x="{margin_left - 20}" y="{y + cell_size / 2}" text-anchor="end" font-size="12" '
            f'font-family="sans-serif">Phase {row}</text>'
        )
        for col in range(2):
            x = margin_left + col * cell_size
            value = matrix[row][col]
            color = STATE_COLORS[col % len(STATE_COLORS)]
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{color}" fill-opacity="0.15" '
                f'stroke="{color}" stroke-width="2" rx="6" ry="6"/>'
            )
            svg_parts.append(
                f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2}" text-anchor="middle" font-size="18" '
                f'font-family="sans-serif">{value:.2f}</text>'
            )

    if dwell is not None and stay < 1.0:
        expected = 1.0 / (1.0 - stay)
        svg_parts.append(
            f'<text x="{plot_width / 2}" y="{plot_height - 60}" text-anchor="middle" font-size="12" '
            f'font-family="sans-serif">Expected dwell ≈ {expected:.1f} steps (target {dwell})</text>'
        )

    svg_parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def posterior_trace_plot(
    posterior: Mapping[str, object],
    out_path: Path,
    *,
    title: str,
    max_steps: int = 200,
) -> None:
    posterior_values = posterior.get("posterior")
    states = posterior.get("states")
    if not isinstance(posterior_values, Sequence) or not posterior_values:
        return
    if not isinstance(states, Sequence) or not states:
        return
    steps = min(len(posterior_values), max_steps, len(states))
    if steps <= 1:
        return

    series = [
        [float(row[state_idx]) for row in posterior_values[:steps]]
        for state_idx in range(len(posterior_values[0]))
    ]
    states_slice = [int(states[i]) for i in range(steps)]

    plot_width = 780
    plot_height = 320
    margin_left, margin_bottom, margin_top, margin_right = 70, 50, 50, 20
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{plot_width / 2}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>',
        f'<text x="{margin_left / 2}" y="{plot_height / 2}" transform="rotate(-90 {margin_left / 2},{plot_height / 2})" '
        'text-anchor="middle" font-size="12" font-family="sans-serif">P(state)</text>',
        f'<line x1="{margin_left}" y1="{plot_height - margin_bottom}" x2="{plot_width - margin_right}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
    ]

    def scale_x(step_idx: int) -> float:
        return margin_left + step_idx / (steps - 1) * inner_width

    def scale_y(value: float) -> float:
        return plot_height - margin_bottom - value * inner_height

    for start, end, state in _state_segments(states_slice):
        x0 = scale_x(start)
        x1 = scale_x(end - 1) if end > start else x0
        width = max(2.0, x1 - x0 + inner_width / max(steps - 1, 1))
        color = STATE_COLORS[state % len(STATE_COLORS)]
        svg_parts.append(
            f'<rect x="{x0:.2f}" y="{margin_top}" width="{width:.2f}" height="{inner_height}" '
            f'fill="{color}" fill-opacity="0.08" stroke="none"/>'
        )

    y_ticks = [i / 4 for i in range(5)]
    for tick in y_ticks:
        y = scale_y(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" stroke="#e0e0e0" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="sans-serif">{tick:.2f}</text>'
        )

    for idx, values in enumerate(series):
        color = STATE_COLORS[idx % len(STATE_COLORS)]
        coords = " ".join(
            f"{scale_x(step):.2f},{scale_y(val):.2f}" for step, val in enumerate(values)
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{coords}"/>'
        )

    legend_x = margin_left + 10
    legend_y = margin_top - 20
    for idx in range(len(series)):
        color = STATE_COLORS[idx % len(STATE_COLORS)]
        svg_parts.append(
            f'<rect x="{legend_x + idx * 120}" y="{legend_y - 12}" width="12" height="12" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + idx * 120 + 18}" y="{legend_y - 2}" font-size="11" font-family="sans-serif">Phase {idx}</text>'
        )

    tick_count = min(10, steps - 1)
    for tick in range(tick_count + 1):
        step = int(round(tick * (steps - 1) / max(1, tick_count)))
        x = scale_x(step)
        svg_parts.append(
            f'<text x="{x:.2f}" y="{plot_height - margin_bottom + 18}" text-anchor="middle" font-size="11" font-family="sans-serif">t={step}</text>'
        )

    svg_parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def agent_mixture_plot(
    mixture: Sequence[Sequence[float]],
    labels: Sequence[str],
    out_path: Path,
    *,
    title: str,
    max_steps: int = 200,
) -> None:
    if not mixture:
        return
    steps = min(len(mixture), max_steps)
    if steps <= 1:
        return
    first_row = mixture[0]
    if not isinstance(first_row, Sequence):
        return
    num_agents = len(first_row)
    if num_agents == 0:
        return

    series = [
        [float(row[agent_idx]) for row in mixture[:steps]]
        for agent_idx in range(num_agents)
    ]

    labels = list(labels)
    if len(labels) != num_agents:
        labels = [f"Agent {idx}" for idx in range(num_agents)]

    plot_width = 780
    plot_height = 320
    margin_left, margin_bottom, margin_top, margin_right = 70, 50, 50, 20
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{plot_width / 2}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>',
        f'<text x="{margin_left / 2}" y="{plot_height / 2}" transform="rotate(-90 {margin_left / 2},{plot_height / 2})" '
        'text-anchor="middle" font-size="12" font-family="sans-serif">Mixture weight</text>',
        f'<line x1="{margin_left}" y1="{plot_height - margin_bottom}" x2="{plot_width - margin_right}" '
        f'y2="{plot_height - margin_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
    ]

    def scale_x(step_idx: int) -> float:
        return margin_left + step_idx / (steps - 1) * inner_width

    def scale_y(value: float) -> float:
        value = max(0.0, min(1.0, float(value)))
        return plot_height - margin_bottom - value * inner_height

    y_ticks = [i / 4 for i in range(5)]
    for tick in y_ticks:
        y = scale_y(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" stroke="#e0e0e0" '
            'stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif">{tick:.2f}</text>'
        )

    for idx, values in enumerate(series):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        coords = " ".join(
            f"{scale_x(step):.2f},{scale_y(val):.2f}" for step, val in enumerate(values)
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{coords}"/>'
        )

    legend_x = margin_left + 10
    legend_y = margin_top - 20
    for idx, label in enumerate(labels):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        svg_parts.append(
            f'<rect x="{legend_x + idx * 140}" y="{legend_y - 12}" width="12" height="12" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + idx * 140 + 18}" y="{legend_y - 2}" font-size="11" font-family="sans-serif">{label}</text>'
        )

    svg_parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _trial_history_design(
    actions: Sequence[int],
    rewards: Sequence[int],
    *,
    max_lag: int = 5,
) -> tuple[List[List[float]], List[float]]:
    if not actions or not rewards:
        return [], []
    length = min(len(actions), len(rewards))
    feats: List[List[float]] = []
    targets: List[float] = []
    for idx in range(max_lag, length):
        row = [1.0]
        for lag in range(1, max_lag + 1):
            choice = 1.0 if actions[idx - lag] == 1 else -1.0
            reward = 1.0 if rewards[idx - lag] > 0 else -1.0
            row.extend([choice, reward, choice * reward])
        feats.append(row)
        targets.append(1.0 if actions[idx] == 1 else 0.0)
    return feats, targets


def _logistic_regression(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    l2: float = 1e-3,
    lr: float = 0.1,
    max_iter: int = 4000,
    tol: float = 1e-6,
) -> List[float]:
    if not features:
        return []
    dim = len(features[0])
    weights = [0.0 for _ in range(dim)]
    n = float(len(features))
    for _ in range(max_iter):
        grads = [0.0 for _ in range(dim)]
        for row, target in zip(features, targets):
            z = sum(w * x for w, x in zip(weights, row))
            p = _sigmoid(z)
            diff = p - target
            for j in range(dim):
                grads[j] += diff * row[j]
        max_update = 0.0
        for j in range(dim):
            grad = grads[j] / n + l2 * weights[j]
            update = lr * grad
            weights[j] -= update
            max_update = max(max_update, abs(update))
        if max_update < tol:
            break
    return weights


def trial_history_coefficients(
    actions: Sequence[int],
    rewards: Sequence[int],
    *,
    max_lag: int = 5,
) -> Optional[Mapping[str, Sequence[float]]]:
    features, targets = _trial_history_design(actions, rewards, max_lag=max_lag)
    if not features:
        return None
    weights = _logistic_regression(features, targets)
    if not weights:
        return None
    coeffs = {
        "choice": [],
        "reward": [],
        "interaction": [],
    }
    idx = 1
    for _ in range(max_lag):
        if idx + 2 >= len(weights):
            break
        coeffs["choice"].append(weights[idx])
        coeffs["reward"].append(weights[idx + 1])
        coeffs["interaction"].append(weights[idx + 2])
        idx += 3
    coeffs["intercept"] = [weights[0]]
    return coeffs


def trial_history_plot(
    runs: Sequence[Mapping[str, object]],
    key: str,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    max_lag: int = 5,
) -> None:
    series: List[tuple] = []
    labels: List[str] = []
    for entry in runs:
        coeffs = entry.get("coeffs")
        if not isinstance(coeffs, Mapping):
            continue
        values = coeffs.get(key)
        if not isinstance(values, Sequence) or not values:
            continue
        labels.append(str(entry.get("label", "Run")))
        series.append(list(values)[:max_lag])
    if not series:
        return

    lags = list(range(1, max(len(values) for values in series) + 1))
    plot_width = 640
    plot_height = 360
    margin_left, margin_bottom, margin_top, margin_right = 70, 50, 40, 20
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    y_min = min(min(values) for values in series)
    y_max = max(max(values) for values in series)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5

    def scale_x(idx: int) -> float:
        return margin_left + idx / max(1, len(lags) - 1) * inner_width

    def scale_y(value: float) -> float:
        return plot_height - margin_bottom - (value - y_min) / (y_max - y_min) * inner_height

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{plot_width / 2}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>',
        f'<text x="{margin_left / 2}" y="{plot_height / 2}" transform="rotate(-90 {margin_left / 2},{plot_height / 2})" '
        f'text-anchor="middle" font-size="12" font-family="sans-serif">{ylabel}</text>',
        f'<line x1="{margin_left}" y1="{plot_height - margin_bottom}" x2="{plot_width - margin_right}" '
        f'y2="{plot_height - margin_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
    ]

    y_ticks = _linear_ticks(y_min, y_max)
    for tick in y_ticks:
        y = scale_y(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" '
            'stroke="#e0e0e0" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif">{_format_float(tick)}</text>'
        )

    for idx, values in enumerate(series):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        coords = " ".join(
            f"{scale_x(lag - 1):.2f},{scale_y(val):.2f}"
            for lag, val in zip(range(1, len(values) + 1), values)
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'
        )
        for lag, val in zip(range(1, len(values) + 1), values):
            svg_parts.append(
                f'<circle cx="{scale_x(lag - 1):.2f}" cy="{scale_y(val):.2f}" r="3" fill="{color}"/>'
            )

    legend_x = margin_left + 10
    legend_y = margin_top + 10
    for idx, label in enumerate(labels):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y + idx * 18}" width="12" height="12" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y + idx * 18 + 10}" font-size="11" '
            f'font-family="sans-serif">{label}</text>'
        )

    for lag in lags:
        x = scale_x(lag - 1)
        svg_parts.append(
            f'<text x="{x:.2f}" y="{plot_height - margin_bottom + 18}" text-anchor="middle" font-size="11" '
            f'font-family="sans-serif">lag {lag}</text>'
        )

    svg_parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def collect_runs(run_dir: Path) -> List[Mapping[str, object]]:
    runs: List[Mapping[str, object]] = []
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        history_path = subdir / "history.json"
        metrics_path = subdir / "metrics.json"
        if not (history_path.exists() and metrics_path.exists()):
            continue
        runs.append(
            {
                "label": subdir.name.replace("_", " ").title(),
                "history": load_history(history_path),
                "metrics": load_metrics(metrics_path),
                "path": subdir,
            }
        )
    if not runs:
        raise FileNotFoundError(
            f"No runs with history.json/metrics.json found under {run_dir}"
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing subfolders for each model (e.g. results/synthetic_run1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fig"),
        help="Directory where SVG figures will be written (default: fig).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix for generated plots (defaults to run directory name).",
    )
    args = parser.parse_args()

    runs = collect_runs(args.run_dir)
    prefix = args.prefix or args.run_dir.name

    sample_path = args.run_dir / "sample_trace.json"
    if sample_path.exists():
        sample_trace = load_sample_trace(sample_path)
        metadata = sample_trace.get("metadata", {}) if isinstance(sample_trace, Mapping) else {}
        trace_body = None
        if isinstance(sample_trace, Mapping):
            trace_body = sample_trace.get("test") or sample_trace.get("train")
        if isinstance(trace_body, Mapping):
            sequence_overview_plot(
                trace_body,
                metadata if isinstance(metadata, Mapping) else {},
                out_path=args.out_dir / f"{prefix}_sequence_overview.svg",
            )
        sticky = metadata.get("sticky") if isinstance(metadata, Mapping) else None
        dwell = metadata.get("dwell") if isinstance(metadata, Mapping) else None
        if sticky is not None:
            transition_matrix_plot(float(sticky), dwell, args.out_dir / f"{prefix}_transition_matrix.svg")

    line_plot(
        runs,
        metric="train_nll",
        title="Training Negative Log-Likelihood",
        ylabel="NLL",
        out_path=args.out_dir / f"{prefix}_train_nll.svg",
    )
    line_plot(
        runs,
        metric="train_accuracy",
        title="Training Action Accuracy",
        ylabel="Accuracy",
        out_path=args.out_dir / f"{prefix}_train_accuracy.svg",
    )
    bar_chart(
        runs,
        metric="phase_accuracy",
        title="Phase Accuracy (Train/Test)",
        ylabel="Accuracy",
        out_path=args.out_dir / f"{prefix}_phase_accuracy.svg",
    )
    bar_chart(
        runs,
        metric="accuracy",
        title="Action Accuracy (Train/Test)",
        ylabel="Accuracy",
        out_path=args.out_dir / f"{prefix}_action_accuracy.svg",
    )

    observed_series = None
    logistic_entries: List[Mapping[str, object]] = []

    for entry in runs:
        run_path = entry.get("path")
        if not isinstance(run_path, Path):
            continue
        posterior_path = run_path / "posterior_trace.json"
        if not posterior_path.exists():
            continue
        posterior_data = load_posterior_trace(posterior_path)
        slug = run_path.name
        posterior_trace_plot(
            posterior_data,
            out_path=args.out_dir / f"{prefix}_{slug}_posterior.svg",
            title=f"{entry.get('label', run_path.name)} Posterior",
        )
        if isinstance(posterior_data, Mapping):
            label_prefix = str(entry.get("label", run_path.name))
            agent_labels = posterior_data.get("agent_labels")
            if not isinstance(agent_labels, Sequence):
                agent_labels = []
            agent_mix = posterior_data.get("agent_mixture")
            if isinstance(agent_mix, Sequence) and agent_mix and isinstance(agent_mix[0], Sequence):
                agent_mixture_plot(
                    [list(map(float, row)) for row in agent_mix],
                    [str(label) for label in agent_labels],
                    out_path=args.out_dir / f"{prefix}_{slug}_agent_mixture.svg",
                    title=f"{label_prefix} Agent Mixture",
                )
            observations = posterior_data.get("observations")
            if isinstance(observations, Mapping):
                obs_actions = observations.get("actions")
                obs_rewards = observations.get("rewards")
                if isinstance(obs_actions, Sequence) and isinstance(obs_rewards, Sequence):
                    actions_seq = [int(a) for a in obs_actions]
                    rewards_seq = [int(r) for r in obs_rewards]
                    if observed_series is None:
                        observed_series = {"actions": actions_seq, "rewards": rewards_seq}
                    policy_logits = posterior_data.get("policy_logits")
                    if isinstance(policy_logits, Sequence):
                        predicted_actions: List[int] = []
                        for row in policy_logits:
                            if isinstance(row, Sequence) and row:
                                best = max(range(len(row)), key=lambda i: row[i])
                                predicted_actions.append(int(best))
                        if predicted_actions:
                            length = min(len(predicted_actions), len(rewards_seq))
                            logistic_entries.append(
                                {
                                    "label": f"{label_prefix} – Model blend",
                                    "actions": predicted_actions[:length],
                                    "rewards": rewards_seq[:length],
                                }
                            )
                    agent_qs = posterior_data.get("agent_Qs")
                    if isinstance(agent_qs, Sequence) and agent_qs:
                        agent_count = 0
                        first_step = agent_qs[0]
                        if isinstance(first_step, Sequence):
                            agent_count = len(first_step)
                        for agent_idx in range(agent_count):
                            agent_actions: List[int] = []
                            for timestep in agent_qs:
                                if not isinstance(timestep, Sequence) or agent_idx >= len(timestep):
                                    agent_actions = []
                                    break
                                q_vals = timestep[agent_idx]
                                if not isinstance(q_vals, Sequence) or not q_vals:
                                    agent_actions = []
                                    break
                                best = max(range(len(q_vals)), key=lambda i: float(q_vals[i]))
                                agent_actions.append(int(best))
                            if not agent_actions:
                                continue
                            length = min(len(agent_actions), len(rewards_seq))
                            if length == 0:
                                continue
                            label = (
                                f"{label_prefix} – {agent_labels[agent_idx]}"
                                if agent_idx < len(agent_labels)
                                else f"{label_prefix} – Agent {agent_idx + 1}"
                            )
                            logistic_entries.append(
                                {
                                    "label": label,
                                    "actions": agent_actions[:length],
                                    "rewards": rewards_seq[:length],
                                }
                            )

    logistic_runs: List[Mapping[str, object]] = []
    if isinstance(observed_series, Mapping):
        coeffs_obs = trial_history_coefficients(
            observed_series.get("actions", []),
            observed_series.get("rewards", []),
        )
        if coeffs_obs:
            logistic_runs.append({"label": "Observed", "coeffs": coeffs_obs})
    for entry in logistic_entries:
        coeffs = trial_history_coefficients(entry.get("actions", []), entry.get("rewards", []))
        if coeffs:
            logistic_runs.append({"label": entry.get("label", "Run"), "coeffs": coeffs})

    if logistic_runs:
        trial_history_plot(
            logistic_runs,
            key="reward",
            title="Trial-history regression (reward)",
            ylabel="Coefficient",
            out_path=args.out_dir / f"{prefix}_trial_history_reward.svg",
        )
        trial_history_plot(
            logistic_runs,
            key="choice",
            title="Trial-history regression (choice)",
            ylabel="Coefficient",
            out_path=args.out_dir / f"{prefix}_trial_history_choice.svg",
        )
        trial_history_plot(
            logistic_runs,
            key="interaction",
            title="Trial-history regression (choice × reward)",
            ylabel="Coefficient",
            out_path=args.out_dir / f"{prefix}_trial_history_interaction.svg",
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
