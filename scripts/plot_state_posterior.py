"""Render SeriesHMM state posteriors as a stacked-area SVG plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence


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


def _format_float(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}" if value >= 1 else f"{value:.3f}"


def _apply_permutation(matrix: Sequence[Sequence[float]], perm: Sequence[int]) -> List[List[float]]:
    if len(matrix) == 0:
        return []
    if len(perm) != len(matrix[0]):
        return [list(row) for row in matrix]
    return [[row[idx] for idx in perm] for row in matrix]


def _stacked_polygon(points_top, points_bottom):
    coords = []
    for x, y in points_top:
        coords.append(f"{x:.2f},{y:.2f}")
    for x, y in reversed(points_bottom):
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def plot_state_posterior(data_path: Path, out_path: Path, title: str) -> None:
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    posterior = payload.get("posterior")
    if not isinstance(posterior, Sequence) or not posterior:
        raise ValueError("posterior_trace.json missing 'posterior' entries")

    if not isinstance(posterior[0], Sequence):
        raise ValueError("posterior entries must be sequences of state probabilities")

    n_states = len(posterior[0])
    n_trials = len(posterior)
    perm = payload.get("best_permutation")
    if isinstance(perm, Sequence):
        posterior = _apply_permutation(posterior, perm)
    else:
        posterior = [list(row) for row in posterior]

    phases = payload.get("phases")
    if isinstance(phases, Sequence) and len(phases) != n_trials:
        phases = None

    plot_width = 640
    plot_height = 420
    margin_left, margin_right = 60, 30
    margin_top, margin_bottom = 70, 100

    axis_bottom = plot_height - margin_bottom
    axis_top = margin_top
    inner_width = plot_width - margin_left - margin_right
    inner_height = axis_bottom - axis_top

    def scale_x(index: int) -> float:
        if n_trials == 1:
            return margin_left + inner_width / 2
        return margin_left + inner_width * index / (n_trials - 1)

    def scale_y(value: float) -> float:
        clamped = min(max(value, 0.0), 1.0)
        return axis_bottom - clamped * inner_height

    # Prepare stacked area coordinates
    top_points: List[List[tuple[float, float]]] = [[] for _ in range(n_states)]
    bottom_points: List[List[tuple[float, float]]] = [[] for _ in range(n_states)]
    for trial_idx, weights in enumerate(posterior):
        if len(weights) != n_states:
            raise ValueError("posterior rows must have consistent length")
        x = scale_x(trial_idx)
        running = 0.0
        for state_idx, weight in enumerate(weights):
            bottom = running
            running += weight
            top = running
            top_points[state_idx].append((x, scale_y(top)))
            bottom_points[state_idx].append((x, scale_y(bottom)))

    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_tick_count = 6
    x_ticks = [1 + i * (n_trials - 1) / (x_tick_count - 1) for i in range(x_tick_count)]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{plot_height}">',
        f'<rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff"/>',
        f'<text x="{plot_width / 2:.1f}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>',
        f'<text x="{margin_left / 2:.1f}" y="{(axis_top + axis_bottom) / 2:.1f}" transform="rotate(-90 {margin_left / 2:.1f},{(axis_top + axis_bottom) / 2:.1f})" text-anchor="middle" font-size="12" font-family="sans-serif">Mixture weight</text>',
        f'<line x1="{margin_left}" y1="{axis_bottom}" x2="{plot_width - margin_right}" y2="{axis_bottom}" stroke="#000"/>',
        f'<line x1="{margin_left}" y1="{axis_top}" x2="{margin_left}" y2="{axis_bottom}" stroke="#000"/>',
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
        x = scale_x(int(round(tick)) - 1)
        svg_parts.append(
            f'<text x="{x:.2f}" y="{axis_bottom + 78:.0f}" text-anchor="middle" font-size="11" font-family="sans-serif">{int(round(tick))}</text>'
        )

    # Annotated phases (if available)
    if phases is not None:
        phase_band_height = 18
        phase_band_y = axis_top - phase_band_height - 10
        bar_width = inner_width / n_trials if n_trials else 0
        svg_parts.append(
            f'<text x="{margin_left}" y="{phase_band_y - 6:.2f}" font-size="12" font-family="sans-serif">Annotated phase</text>'
        )
        for trial_idx, phase in enumerate(phases):
            try:
                phase_int = int(phase)
            except (TypeError, ValueError):
                continue
            color = COLOR_PALETTE[phase_int % len(COLOR_PALETTE)]
            x = margin_left + trial_idx * (inner_width / max(n_trials, 1))
            svg_parts.append(
                f'<rect x="{x:.2f}" y="{phase_band_y:.2f}" width="{bar_width:.2f}" height="{phase_band_height}" fill="{color}" opacity="0.35"/>'
            )

    # Dominant state band
    dominant_band_y = axis_bottom + 20
    dominant_band_height = 40
    bar_width = inner_width / n_trials if n_trials else 0
    svg_parts.append(
        f'<text x="{margin_left}" y="{dominant_band_y - 8:.2f}" font-size="12" font-family="sans-serif">Dominant state</text>'
    )
    for trial_idx, weights in enumerate(posterior):
        dominant = max(range(n_states), key=lambda idx: weights[idx])
        color = COLOR_PALETTE[dominant % len(COLOR_PALETTE)]
        x = margin_left + trial_idx * (inner_width / max(n_trials, 1))
        svg_parts.append(
            f'<rect x="{x:.2f}" y="{dominant_band_y:.2f}" width="{bar_width:.2f}" height="{dominant_band_height}" fill="{color}" opacity="0.35"/>'
        )

    # Stacked areas and outlines
    for state_idx in range(n_states):
        color = COLOR_PALETTE[state_idx % len(COLOR_PALETTE)]
        polygon_points = _stacked_polygon(top_points[state_idx], bottom_points[state_idx])
        svg_parts.append(
            f'<polygon points="{polygon_points}" fill="{color}" opacity="0.45" stroke="none"/>'
        )
        coords = " ".join(
            f"{x:.2f},{y:.2f}"
            for (x, y) in top_points[state_idx]
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{coords}"/>'
        )

    legend_x = plot_width - margin_right - 140
    legend_y = axis_top + 10
    for state_idx in range(n_states):
        color = COLOR_PALETTE[state_idx % len(COLOR_PALETTE)]
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y + state_idx * 18}" width="12" height="12" fill="{color}" opacity="0.8"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y + state_idx * 18 + 10}" font-size="11" font-family="sans-serif">State {state_idx + 1}</text>'
        )

    svg_parts.append(
        f'<text x="{plot_width / 2:.1f}" y="{plot_height - 12}" text-anchor="middle" font-size="12" font-family="sans-serif">Trial index</text>'
    )
    svg_parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SeriesHMM state posteriors as an SVG figure.")
    parser.add_argument("data", type=Path, help="Path to posterior_trace.json")
    parser.add_argument("out", type=Path, help="Output SVG path")
    parser.add_argument(
        "--title",
        default="SeriesHMM state posterior",
        help="Title to place at the top of the figure",
    )
    args = parser.parse_args()

    plot_state_posterior(args.data, args.out, args.title)


if __name__ == "__main__":
    main()
