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
import re
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

COLOR_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#FF9DA6",
    "#9C755F",
    "#79706E",
]


def load_history(path: Path) -> Sequence[Mapping[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics(path: Path) -> Mapping[str, Mapping[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_trial_history(path: Path) -> Mapping[str, object]:
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


def _slugify(label: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", label.lower()).strip("_")
    return cleaned or "series"


def plot_trial_history_panel(entry: Mapping[str, object], *, lags: int, out_path: Path) -> None:
    combos = [
        ("common_reward", "Common reward", "#E45756", None),
        ("common_omission", "Common omission", "#E45756", "6 3"),
        ("rare_reward", "Rare reward", "#4C78A8", None),
        ("rare_omission", "Rare omission", "#4C78A8", "6 3"),
    ]

    values = []
    for key, _, _, _ in combos:
        seq = entry.get(key)
        if isinstance(seq, Sequence):
            values.append([float(v) for v in seq])
        else:
            values.append([])

    if not any(series for series in values):
        return

    plot_width = 480
    plot_height = 320
    margin_left, margin_right = 70, 40
    margin_top, margin_bottom = 45, 60
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    if lags <= 0:
        return

    if lags == 1:
        x_coords = [margin_left + inner_width / 2]
    else:
        x_coords = [margin_left + inner_width * idx / (lags - 1) for idx in range(lags)]

    abs_max = 0.0
    for series in values:
        for coeff in series:
            abs_max = max(abs_max, abs(coeff))
    if math.isclose(abs_max, 0.0):
        abs_max = 1.0
    y_max = abs_max * 1.05
    y_min = -y_max

    def scale_y(value: float) -> float:
        return plot_height - margin_bottom - (value - y_min) / (y_max - y_min) * inner_height

    y_ticks = _linear_ticks(y_min, y_max)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        f'<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{plot_width / 2}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">{entry.get("label", "Series")}</text>',
        f'<text x="{margin_left / 2}" y="{plot_height / 2}" transform="rotate(-90 {margin_left / 2},{plot_height / 2})" text-anchor="middle" font-size="12" font-family="sans-serif">Stay log-odds</text>',
        f'<text x="{plot_width / 2}" y="{plot_height - 20}" text-anchor="middle" font-size="12" font-family="sans-serif">Trials back</text>',
        f'<line x1="{margin_left}" y1="{plot_height - margin_bottom}" x2="{plot_width - margin_right}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_height - margin_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{scale_y(0):.2f}" x2="{plot_width - margin_right}" y2="{scale_y(0):.2f}" stroke="#bbbbbb"/>',
    ]

    for tick in y_ticks:
        y = scale_y(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" stroke="#e0e0e0" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="sans-serif">{_format_float(tick)}</text>'
        )

    for idx, (key, _, color, dash) in enumerate(combos):
        coeffs = values[idx]
        if not coeffs:
            continue
        coords = " ".join(f"{x_coords[i]:.2f},{scale_y(val):.2f}" for i, val in enumerate(coeffs))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} points="{coords}"/>'
        )
        for i, val in enumerate(coeffs):
            svg_parts.append(
                f'<circle cx="{x_coords[i]:.2f}" cy="{scale_y(val):.2f}" r="3" fill="{color}"/>'
            )

    legend_x = plot_width - margin_right + 5
    legend_y = margin_top
    for idx, (_, legend_label, color, dash) in enumerate(combos):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg_parts.append(
            f'<line x1="{legend_x}" y1="{legend_y + idx * 18}" x2="{legend_x + 18}" y2="{legend_y + idx * 18}" stroke="{color}" stroke-width="2"{dash_attr}/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 22}" y="{legend_y + idx * 18 + 4}" font-size="11" font-family="sans-serif">{legend_label}</text>'
        )

    for i, x in enumerate(x_coords):
        svg_parts.append(
            f'<text x="{x:.2f}" y="{plot_height - margin_bottom + 16}" text-anchor="middle" font-size="11" font-family="sans-serif">{i + 1}</text>'
        )

    svg_parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def _compute_agent_mix(posterior_entry: Mapping[str, object]) -> List[List[float]]:
    posterior = posterior_entry.get("posterior")
    gating = posterior_entry.get("gating")
    if not (isinstance(posterior, Sequence) and isinstance(gating, Sequence)):
        return []

    mixture: List[List[float]] = []
    for t, (gamma_t, gate_t) in enumerate(zip(posterior, gating)):
        if not (isinstance(gamma_t, Sequence) and isinstance(gate_t, Sequence)):
            continue
        if gate_t and isinstance(gate_t[0], Sequence):
            weights = [0.0 for _ in range(len(gate_t[0]))]
        else:
            weights = []
        for state_idx, gamma_val in enumerate(gamma_t):
            if state_idx >= len(gate_t):
                break
            gate_state = gate_t[state_idx]
            if not isinstance(gate_state, Sequence):
                continue
            for agent_idx, gate_val in enumerate(gate_state):
                if agent_idx >= len(weights):
                    weights.extend([0.0] * (agent_idx + 1 - len(weights)))
                weights[agent_idx] += float(gamma_val) * float(gate_val)
        if weights:
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
        if weights:
            mixture.append(weights)
    return mixture


def _solve_weight_system(rows: Sequence[Sequence[float]], rhs: Sequence[float], *, ridge: float = 1e-3) -> List[float]:
    if not rows:
        return []
    agents = len(rows[0])
    if agents == 0:
        return []
    normal = [[0.0 for _ in range(agents)] for _ in range(agents)]
    targets = [0.0 for _ in range(agents)]
    for row, value in zip(rows, rhs):
        if len(row) != agents:
            continue
        for i in range(agents):
            targets[i] += row[i] * value
            for j in range(agents):
                normal[i][j] += row[i] * row[j]
    for i in range(agents):
        normal[i][i] += ridge

    aug = [normal[i][:] + [targets[i]] for i in range(agents)]
    for i in range(agents):
        pivot_row = max(range(i, agents), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot_row][i]) < 1e-9:
            return []
        if pivot_row != i:
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]
        pivot = aug[i][i]
        for j in range(i, agents + 1):
            aug[i][j] /= pivot
        for r in range(agents):
            if r == i:
                continue
            factor = aug[r][i]
            for c in range(i, agents + 1):
                aug[r][c] -= factor * aug[i][c]
    weights = [aug[i][agents] for i in range(agents)]
    weights = [max(0.0, w) for w in weights]
    total = sum(weights)
    if total <= 1e-8:
        if agents == 0:
            return []
        return [1.0 / agents for _ in range(agents)]
    return [w / total for w in weights]


def _infer_agent_projection(posterior_entry: Mapping[str, object]) -> List[List[float]]:
    logits = posterior_entry.get("pi_log")
    baseline = posterior_entry.get("baseline_q")
    if not (isinstance(logits, Sequence) and isinstance(baseline, Sequence)):
        return []

    mixture: List[List[float]] = []
    for log_row, q_row in zip(logits, baseline):
        if not (isinstance(log_row, Sequence) and isinstance(q_row, Sequence)):
            continue
        if len(log_row) < 2:
            continue
        coeffs: List[float] = []
        for agent_vals in q_row:
            if isinstance(agent_vals, Sequence) and len(agent_vals) >= 2:
                coeffs.append(float(agent_vals[1]) - float(agent_vals[0]))
            else:
                coeffs.append(0.0)
        if not coeffs:
            continue
        preference = float(log_row[1]) - float(log_row[0])
        weights = _solve_weight_system([coeffs, [1.0 for _ in coeffs]], [preference, 1.0])
        if weights:
            mixture.append(weights)
    return mixture


def plot_mixture_dynamics(
    values: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    title: str,
    band_label: str,
    out_path: Path,
    ground_truth: Sequence[int] | None = None,
    ground_truth_label: str | None = None,
    palette: Sequence[str] | None = None,
) -> None:
    if not values:
        return

    trials = len(values)
    agents = len(values[0]) if values[0] else 0
    if trials == 0 or agents == 0:
        return

    plot_width = 640
    plot_height = 460 if ground_truth else 420
    margin_left = 60
    margin_right = 30
    margin_top = 40
    margin_bottom = 60
    band_height = 40
    band_gap = 20
    gt_band_height = 28 if ground_truth else 0
    total_band = band_height + (gt_band_height + band_gap if ground_truth else 0)

    inner_width = plot_width - margin_left - margin_right
    main_bottom = margin_bottom + total_band
    inner_height = plot_height - margin_top - main_bottom
    if inner_height <= 0:
        inner_height = plot_height * 0.5

    def scale_x(index: int) -> float:
        if trials <= 1:
            return margin_left + inner_width / 2
        return margin_left + inner_width * index / (trials - 1)

    def scale_y(value: float) -> float:
        return plot_height - main_bottom - value * inner_height

    x_ticks = _linear_ticks(1, trials, count=min(6, trials))
    y_ticks = _linear_ticks(0.0, 1.0)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        f'<rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff"/>',
        f'<text x="{plot_width / 2}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>',
        f'<text x="{margin_left / 2}" y="{margin_top + inner_height / 2}" transform="rotate(-90 {margin_left / 2},{margin_top + inner_height / 2})" text-anchor="middle" font-size="12" font-family="sans-serif">Mixture weight</text>',
        f'<line x1="{margin_left}" y1="{plot_height - main_bottom}" x2="{plot_width - margin_right}" y2="{plot_height - main_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_height - main_bottom}" stroke="#000"/>',
    ]

    for tick in y_ticks:
        y = scale_y(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" stroke="#e0e0e0" stroke-dasharray="4 4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="sans-serif">{_format_float(tick)}</text>'
        )

    for tick in x_ticks:
        idx = max(0, min(trials - 1, int(round(tick - 1))))
        x = scale_x(idx)
        svg_parts.append(
            f'<text x="{x:.2f}" y="{plot_height - margin_bottom + 18}" text-anchor="middle" font-size="11" font-family="sans-serif">{_format_float(tick)}</text>'
        )

    palette = list(palette or COLOR_PALETTE)

    for agent_idx in range(agents):
        color = palette[agent_idx % len(palette)]
        coords = " ".join(
            f"{scale_x(t):.2f},{scale_y(values[t][agent_idx]):.2f}"
            for t in range(trials)
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'
        )
        for t in range(trials):
            svg_parts.append(
                f'<circle cx="{scale_x(t):.2f}" cy="{scale_y(values[t][agent_idx]):.2f}" r="2.5" fill="{color}"/>'
            )

    legend_x = plot_width - margin_right - 120
    legend_y = margin_top
    for idx, label in enumerate(labels):
        color = palette[idx % len(palette)]
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y + idx * 18}" width="12" height="12" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y + idx * 18 + 10}" font-size="11" font-family="sans-serif">{label}</text>'
        )

    dominant = [max(range(agents), key=lambda idx: values[t][idx]) for t in range(trials)]
    band_top = plot_height - margin_bottom - band_height
    band_width = inner_width / max(1, trials)
    for t, agent_idx in enumerate(dominant):
        color = palette[agent_idx % len(palette)]
        x = margin_left + band_width * t
        svg_parts.append(
            f'<rect x="{x:.2f}" y="{band_top:.2f}" width="{band_width + 0.5:.2f}" height="{band_height:.2f}" fill="{color}" opacity="0.35"/>'
        )

    svg_parts.append(
        f'<rect x="{margin_left}" y="{band_top - 22:.2f}" width="160" height="18" fill="none" stroke="none"/>'
    )
    svg_parts.append(
        f'<text x="{margin_left}" y="{band_top - 8:.2f}" font-size="12" font-family="sans-serif">{band_label}</text>'
    )

    if ground_truth:
        gt_band_top = band_top - (gt_band_height + band_gap)
        ground_truth_label = ground_truth_label or "Ground truth"
        for t, state_idx in enumerate(ground_truth):
            color = palette[state_idx % len(palette)]
            x = margin_left + band_width * t
            svg_parts.append(
                f'<rect x="{x:.2f}" y="{gt_band_top:.2f}" width="{band_width + 0.5:.2f}" height="{gt_band_height:.2f}" fill="{color}" opacity="0.6"/>'
            )
        svg_parts.append(
            f'<text x="{margin_left}" y="{gt_band_top - 8:.2f}" font-size="12" font-family="sans-serif">{ground_truth_label}</text>'
        )

    svg_parts.append(
        f'<text x="{margin_left + inner_width / 2}" y="{plot_height - margin_bottom + 40}" text-anchor="middle" font-size="12" font-family="sans-serif">Trial index</text>'
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
        entry = {
            "label": subdir.name.replace("_", " ").title(),
            "slug": subdir.name,
            "history": load_history(history_path),
            "metrics": load_metrics(metrics_path),
        }
        posterior_path = subdir / "posterior_trace.json"
        if posterior_path.exists():
            entry["posterior"] = load_posterior_trace(posterior_path)
        runs.append(entry)
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

    trial_history_path = args.run_dir / "trial_history.json"
    if trial_history_path.exists():
        history = load_trial_history(trial_history_path)
        series = history.get("series")
        lags = int(history.get("lags", 0))
        if isinstance(series, Sequence) and lags > 0:
            for entry in series:
                if not isinstance(entry, Mapping):
                    continue
                label = str(entry.get("label", "Series"))
                slug = _slugify(label)
                out_path = args.out_dir / f"{prefix}_trial_history_{slug}.svg"
                plot_trial_history_panel(entry, lags=lags, out_path=out_path)

    agent_defaults = ["MF Reward", "MF Choice", "Model-based", "Bias"]
    for run in runs:
        posterior_entry = run.get("posterior")
        if not isinstance(posterior_entry, Mapping):
            continue
        label = str(posterior_entry.get("label", run.get("label", "Run")))
        slug_key = str(run.get("slug", label))
        slug = _slugify(slug_key)

        posterior = posterior_entry.get("posterior")
        values: List[List[float]] = []
        state_labels: List[str] = []
        gt_states: Sequence[int] | None = None
        if isinstance(posterior, Sequence) and posterior:
            first = posterior[0]
            if isinstance(first, Sequence) and first:
                state_count = len(first)
                permutation = posterior_entry.get("best_permutation")
                for row in posterior:
                    if not isinstance(row, Sequence):
                        continue
                    row_vals = [float(val) for val in row[:state_count]]
                    values.append(row_vals)
                if permutation and isinstance(permutation, Sequence) and len(permutation) == state_count:
                    permuted: List[List[float]] = []
                    for row in values:
                        permuted.append([row[idx] for idx in permutation])
                    values = permuted
                state_labels = [f"State {idx + 1}" for idx in range(state_count)]
                phases = posterior_entry.get("phases")
                if isinstance(phases, Sequence) and phases:
                    try:
                        gt_states = [int(p) for p in phases]
                    except (TypeError, ValueError):
                        gt_states = None

        gating_mix = _compute_agent_mix(posterior_entry)
        projected_mix = _infer_agent_projection(posterior_entry)
        if gating_mix:
            agent_count = len(gating_mix[0])
            if agent_count == len(agent_defaults):
                labels = agent_defaults
            else:
                labels = [f"Agent {idx + 1}" for idx in range(agent_count)]
            plot_mixture_dynamics(
                gating_mix,
                labels,
                title=f"{label} agent responsibilities",
                band_label="Dominant agent",
                out_path=args.out_dir / f"{prefix}_agent_mix_{slug}.svg",
                ground_truth=gt_states,
                ground_truth_label="Ground truth phase" if gt_states else None,
            )
        elif projected_mix:
            agent_count = len(projected_mix[0]) if projected_mix[0] else 0
            if agent_count == len(agent_defaults):
                labels = agent_defaults
            else:
                labels = [f"Agent {idx + 1}" for idx in range(agent_count)]
            plot_mixture_dynamics(
                projected_mix,
                labels,
                title=f"{label} agent projection",
                band_label="Dominant agent (projected)",
                out_path=args.out_dir / f"{prefix}_agent_mix_{slug}.svg",
                ground_truth=gt_states,
                ground_truth_label="Ground truth phase" if gt_states else None,
            )
        elif values:
            plot_mixture_dynamics(
                values,
                state_labels,
                title=f"{label} phase responsibilities",
                band_label="Dominant phase",
                out_path=args.out_dir / f"{prefix}_agent_mix_{slug}.svg",
                ground_truth=gt_states,
                ground_truth_label="Ground truth phase" if gt_states else None,
                palette=COLOR_PALETTE,
            )

        if values:
            plot_mixture_dynamics(
                values,
                state_labels,
                title=f"{label} state posterior",
                band_label="Dominant state",
                out_path=args.out_dir / f"{prefix}_state_posterior_{slug}.svg",
                ground_truth=gt_states,
                ground_truth_label="Ground truth state" if gt_states else None,
                palette=COLOR_PALETTE,
            )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
