"""Train TinyMoARNN on synthetic HMM-MoA data and export illustrative artefacts.

The command mirrors the configuration used in :mod:`run_series_experiment`
but focuses on the recurrent-only baseline. After fitting, a single test
trajectory can be serialised to JSON and/or plotted so that downstream
analyses (or reports) can visualise how the RNN's action probabilities,
agent gates, and latent phases evolve over time. Optional training-history
exports make it easy to track optimisation progress as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import torch

from . import TinyMoARNN, eval_epoch_tiny, train_epoch_tiny, two_step_mb_generator
from .trial_history import default_agent_suite


def _default_agents():
    return default_agent_suite()


def _export_trace(
    path: Path,
    agent_names: Sequence[str],
    actions: torch.Tensor,
    rewards: torch.Tensor,
    transitions: torch.Tensor,
    latent_state: torch.Tensor,
    pi_log: torch.Tensor,
    gates: torch.Tensor,
    q_seq: torch.Tensor,
) -> None:
    """Serialise an evaluation trajectory to JSON for visualisation."""

    probs = pi_log.exp().cpu()
    gates = gates.cpu()
    q_seq = q_seq.cpu()
    actions = actions.cpu()
    rewards = rewards.cpu()
    transitions = transitions.cpu()
    latent_state = latent_state.cpu()

    T = actions.size(0)
    trace: List[Mapping[str, object]] = []
    for t in range(T):
        trace.append(
            {
                "t": int(t),
                "action": int(actions[t].item()),
                "reward": int(rewards[t].item()),
                "transition": int(transitions[t].item()),
                "latent_state": int(latent_state[t].item()),
                "action_probs": {
                    "stay": float(probs[t, 0].item()),
                    "switch": float(probs[t, 1].item()),
                },
                "agent_gates": {
                    name: float(gates[t, i].item())
                    for i, name in enumerate(agent_names)
                },
                "agent_q": {
                    name: {
                        "stay": float(q_seq[t, i, 0].item()),
                        "switch": float(q_seq[t, i, 1].item()),
                    }
                    for i, name in enumerate(agent_names)
                },
            }
        )

    payload = {
        "summary": {
            "T": T,
            "agent_names": list(agent_names),
        },
        "trace": trace,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _export_history(path: Path, history: Sequence[Tuple[float, float]]) -> None:
    """Write the per-epoch training metrics to JSON."""

    payload = [
        {"epoch": idx + 1, "train_nll": loss, "train_acc": acc}
        for idx, (loss, acc) in enumerate(history)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _plot_trace(
    path: Path,
    agent_names: Sequence[str],
    actions: torch.Tensor,
    rewards: torch.Tensor,
    transitions: torch.Tensor,
    latent_state: torch.Tensor,
    pi_log: torch.Tensor,
    gates: torch.Tensor,
) -> bool:
    """Render a multi-panel figure visualising a fitted trajectory."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print(
            "matplotlib is not installed; skipping plot export",
            file=sys.stderr,
        )
        return False

    probs = pi_log.exp().cpu().numpy()
    gates_np = gates.cpu().numpy()
    actions_np = actions.cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    transitions_np = transitions.cpu().numpy()
    latent_np = latent_state.cpu().numpy()

    T = actions_np.shape[0]
    steps = range(T)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    ax_prob = axes[0]
    ax_prob.plot(steps, probs[:, 0], label="P(stay)", color="#3366cc")
    ax_prob.plot(steps, probs[:, 1], label="P(switch)", color="#dc3912")
    ax_prob.scatter(
        steps,
        actions_np,
        c="#109618",
        s=16,
        alpha=0.8,
        label="chosen action",
    )
    ax_prob.set_ylabel("Action prob.")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend(loc="upper right", frameon=False)

    ax_gate = axes[1]
    for idx, name in enumerate(agent_names):
        ax_gate.plot(steps, gates_np[:, idx], label=name)
    ax_gate.set_ylabel("Gate weight")
    ax_gate.set_ylim(-0.05, 1.05)
    ax_gate.legend(loc="upper right", frameon=False, ncol=min(2, len(agent_names)))

    ax_latent = axes[2]
    ax_latent.step(steps, latent_np, where="post", label="latent state")
    ax_latent.scatter(steps, rewards_np, marker="x", s=12, c="#ff9900", label="reward")
    ax_latent.scatter(
        steps,
        transitions_np,
        marker="s",
        s=12,
        facecolors="none",
        edgecolors="#0099c6",
        label="transition",
    )
    ax_latent.set_xlabel("Time step")
    ax_latent.set_ylabel("Latent / events")
    ax_latent.legend(loc="upper right", frameon=False)

    fig.suptitle("TinyMoARNN fit on synthetic HMM-MoA data")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def run(
    *,
    epochs: int = 150,
    B: int = 64,
    T: int = 400,
    hidden: int = 2,
    lr: float = 1e-3,
    seed: int = 1,
    dwell: int = 120,
    beta: float = 3.5,
    device: str | None = None,
    trace_index: int = 0,
    trace_out: str | None = None,
    plot_out: str | None = None,
    history_out: str | None = None,
) -> Mapping[str, object]:
    """Train TinyMoARNN and optionally export a representative trace."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agents = _default_agents()
    agent_names = [name for name, _ in agents]

    a_tr, r_tr, t_tr, _ = two_step_mb_generator(
        B=B, T=T, dwell=dwell, beta=beta, seed=seed, device=device
    )
    a_te, r_te, t_te, s_te = two_step_mb_generator(
        B=B, T=T, dwell=dwell, beta=beta, seed=seed + 1, device=device
    )

    model = TinyMoARNN(n_agents=len(agents), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Tuple[float, float]] = []
    for _ in range(epochs):
        loss, acc = train_epoch_tiny(model, opt, a_tr, r_tr, t_tr, agents)
        history.append((loss, acc))

    test_loss, test_acc, pi_log, gates, q_seq = eval_epoch_tiny(
        model, a_te, r_te, t_te, agents, return_details=True
    )
    metrics = {"test_nll": test_loss, "test_acc": test_acc}

    trace_written = False
    plot_written = False
    history_written = False

    if trace_out is not None or plot_out is not None:
        idx = max(0, min(trace_index, a_te.size(0) - 1))
        if trace_out is not None:
            _export_trace(
                Path(trace_out),
                agent_names,
                a_te[idx],
                r_te[idx],
                t_te[idx],
                s_te[idx],
                pi_log[idx],
                gates[idx],
                q_seq[idx],
            )
            trace_written = True
        if plot_out is not None:
            plot_written = _plot_trace(
                Path(plot_out),
                agent_names,
                a_te[idx],
                r_te[idx],
                t_te[idx],
                s_te[idx],
                pi_log[idx],
                gates[idx],
            )

    if history_out is not None:
        _export_history(Path(history_out), history)
        history_written = True

    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "artefacts": {
            "trace": trace_written,
            "plot": plot_written,
            "history": history_written,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dwell", type=int, default=120)
    parser.add_argument("--beta", type=float, default=3.5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--trace-out",
        default="fig/tiny_trace.json",
        help="Where to write the exported evaluation trace (set to '' to skip)",
    )
    parser.add_argument(
        "--plot-out",
        default="fig/tiny_trace.png",
        help="Optional path for a plotted evaluation trace (set to '' to skip)",
    )
    parser.add_argument(
        "--history-out",
        default="fig/tiny_history.json",
        help="Where to dump per-epoch training metrics (set to '' to skip)",
    )
    parser.add_argument(
        "--trace-index",
        type=int,
        default=0,
        help="Which evaluation sequence to export",
    )

    args = parser.parse_args(argv)

    trace_out = args.trace_out if args.trace_out else None
    plot_out = args.plot_out if args.plot_out else None
    history_out = args.history_out if args.history_out else None
    result = run(
        epochs=args.epochs,
        B=args.B,
        T=args.T,
        hidden=args.hidden,
        lr=args.lr,
        seed=args.seed,
        dwell=args.dwell,
        beta=args.beta,
        device=args.device,
        trace_index=args.trace_index,
        trace_out=trace_out,
        plot_out=plot_out,
        history_out=history_out,
    )

    metrics = result["metrics"]
    print(f"TinyMoARNN -> NLL: {metrics['test_nll']:.3f}  Acc: {metrics['test_acc']:.3f}")
    artefacts = result["artefacts"]
    if trace_out and artefacts["trace"]:
        print(f"Exported evaluation trace to {trace_out}")
    if plot_out and artefacts["plot"]:
        print(f"Exported evaluation plot to {plot_out}")
    if history_out and artefacts["history"]:
        print(f"Saved training history to {history_out}")


if __name__ == "__main__":
    main()
