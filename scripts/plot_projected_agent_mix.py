"""Render the projected TinyRNN agent responsibilities as an SVG line plot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

COLOR_PALETTE = [
    "#4C78A8",  # MF Reward
    "#F58518",  # MF Choice
    "#54A24B",  # Model-based
    "#E45756",  # Bias
    "#72B7B2",
    "#FF9DA6",
    "#9C755F",
    "#79706E",
]


def _format_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{int(round(value))}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot_projected_mix(data_path: Path, out_path: Path, title: str) -> None:
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    agent_names = payload.get("agent_names")
    projected_mix = payload.get("projected_mix")
    if not isinstance(agent_names, Sequence) or not isinstance(projected_mix, Sequence):
        raise ValueError("projected_agent_mix.json is missing required fields")

    n_agents = len(agent_names)
    if n_agents == 0:
        raise ValueError("no agents available to plot")
    if any(len(row) != n_agents for row in projected_mix):
        raise ValueError("each projected_mix row must contain one value per agent")

    n_trials = len(projected_mix)
    if n_trials == 0:
        raise ValueError("projected_mix must contain at least one trial")

    plot_width = 640
    plot_height = 420
    margin_left, margin_right = 60, 30
    margin_top, margin_bottom = 40, 100

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

    dominant_band_y = axis_bottom + 20
    dominant_band_height = 40
    bar_width = inner_width / n_trials

    for trial_idx, weights in enumerate(projected_mix):
        dominant = max(range(n_agents), key=lambda idx: weights[idx])
        color = COLOR_PALETTE[dominant % len(COLOR_PALETTE)]
        x = margin_left + trial_idx * bar_width
        svg_parts.append(
            f'<rect x="{x:.2f}" y="{dominant_band_y:.2f}" width="{bar_width:.2f}" height="{dominant_band_height}" fill="{color}" opacity="0.35"/>'
        )

    for agent_idx, agent_name in enumerate(agent_names):
        color = COLOR_PALETTE[agent_idx % len(COLOR_PALETTE)]
        coords = " ".join(
            f"{scale_x(trial_idx):.2f},{scale_y(weights[agent_idx]):.2f}"
            for trial_idx, weights in enumerate(projected_mix)
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'
        )
        for trial_idx, weights in enumerate(projected_mix):
            svg_parts.append(
                f'<circle cx="{scale_x(trial_idx):.2f}" cy="{scale_y(weights[agent_idx]):.2f}" r="2.5" fill="{color}"/>'
            )

    legend_x = plot_width - margin_right - 150
    legend_y = axis_top
    for agent_idx, agent_name in enumerate(agent_names):
        color = COLOR_PALETTE[agent_idx % len(COLOR_PALETTE)]
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y + agent_idx * 18}" width="12" height="12" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y + agent_idx * 18 + 10}" font-size="11" font-family="sans-serif">{agent_name}</text>'
        )

    svg_parts.append(
        f'<text x="{margin_left}" y="{dominant_band_y - 8}" font-size="12" font-family="sans-serif">Dominant agent</text>'
    )
    svg_parts.append(
        f'<text x="{plot_width / 2:.1f}" y="{plot_height - 12}" text-anchor="middle" font-size="12" font-family="sans-serif">Trial index</text>'
    )
    svg_parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the projected TinyRNN agent mix as an SVG line chart.")
    parser.add_argument("data", type=Path, help="Path to projected_agent_mix.json")
    parser.add_argument("out", type=Path, help="Path to write the SVG output")
    parser.add_argument(
        "--title",
        default="Projected TinyRNN agent responsibilities",
        help="Title to place at the top of the figure",
    )
    args = parser.parse_args()

    plot_projected_mix(args.data, args.out, args.title)


if __name__ == "__main__":
    main()
