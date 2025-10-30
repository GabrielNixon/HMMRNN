"""Pipeline for fitting SeriesHMM models on external datasets.

The command expects a preprocessed `.npz` bundle that contains one or more
behavioural sessions from the Mixture-of-Agents repository.  Each array should
be shaped ``[num_sessions, T]`` where ``T`` is the shared number of trials per
session.  The file must provide integer-valued ``actions`` (0/1), ``rewards``
and ``transitions`` indicators.  If latent phase annotations are available the
``phases`` key can be supplied as well; otherwise the scripts will simply omit
phase-accuracy metrics.

Because the execution environment used for automated evaluation cannot reach
the MixtureAgentsModels GitHub repository, this module also exposes a
``--demo-synthetic`` switch so automated tests can run the pipeline end-to-end
on a locally generated surrogate dataset.  When real data are available the
flag should be omitted and ``--data`` should point at the converted ``.npz``
file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import torch

from .data import two_step_mb_generator
from .metrics import phase_accuracy_permuted, align_gamma
from .models import SeriesHMMTinyMoARNN, SeriesHMMTinyRNN
from .train import eval_epoch_series, train_epoch_series
from .trial_history import (
    agent_action_sequences,
    default_agent_suite,
    summarise_trial_history,
)


def init_sticky(series_model, stay=0.97, eps=1e-3):
    with torch.no_grad():
        A = torch.tensor([[stay, 1 - stay], [1 - stay, stay]], device=series_model.hmm.log_A.device)
        series_model.hmm.log_A.copy_(torch.log(A))
        for param in (series_model.hmm.log_pi,):
            param.add_(eps * torch.randn_like(param))
        if hasattr(series_model, "by"):
            series_model.by.add_(eps * torch.randn_like(series_model.by))
        if hasattr(series_model, "Wg"):
            series_model.Wg.add_(eps * torch.randn_like(series_model.Wg))
        if hasattr(series_model, "head"):
            series_model.head.weight.add_(eps * torch.randn_like(series_model.head.weight))
            if series_model.head.bias is not None:
                series_model.head.bias.add_(eps * torch.randn_like(series_model.head.bias))


def load_npz_dataset(path: Path, *, device: str = "cpu") -> Dict[str, torch.Tensor]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime extras
        raise SystemExit(
            "NumPy is required to read .npz behavioural bundles. Install numpy locally and rerun."
        ) from exc

    arrays = np.load(path, allow_pickle=True)
    required = {"actions", "rewards", "transitions"}
    missing = required - set(arrays.keys())
    if missing:
        raise ValueError(f"dataset {path} missing required arrays: {sorted(missing)}")

    def _to_tensor(key: str, dtype) -> torch.Tensor:
        data = arrays[key]
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError(f"array '{key}' should be rank-2 (sessions x T), found shape {data.shape}")
        return torch.as_tensor(data, dtype=dtype, device=device)

    payload: Dict[str, torch.Tensor] = {
        "actions": _to_tensor("actions", torch.long),
        "rewards": _to_tensor("rewards", torch.long),
        "transitions": _to_tensor("transitions", torch.long),
    }
    if "phases" in arrays:
        payload["phases"] = _to_tensor("phases", torch.long)
    return payload


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def tensor_first_sequence(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()[0]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def evaluate_series(model, batch, agents=None):
    loss, acc, gk, lg, pi_log = eval_epoch_series(
        model,
        batch["actions"],
        batch["rewards"],
        batch["transitions"],
        agents=agents,
    )
    gamma = torch.softmax(lg, dim=-1)
    metrics = {"nll": float(loss), "accuracy": float(acc)}
    extras = {"gamma": gamma, "gating": gk, "pi_log": pi_log}
    if "phases" in batch:
        phase_acc, perm, confusion = phase_accuracy_permuted(gamma, batch["phases"])
        metrics.update(
            {
                "phase_accuracy": float(phase_acc),
                "best_permutation": list(perm),
                "confusion": confusion.cpu().tolist(),
            }
        )
        extras["aligned_gamma"] = align_gamma(gamma, perm)
    return metrics, extras


def dump_training_artifacts(out_dir: Path, model, history, eval_results, *, save_checkpoint=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(eval_results, indent=2))
    if save_checkpoint:
        torch.save({"state_dict": model.state_dict()}, out_dir / "model.pt")


def build_history(epoch_history):
    return [
        {"epoch": epoch + 1, "train_nll": float(loss), "train_accuracy": float(acc)}
        for epoch, (loss, acc) in enumerate(epoch_history)
    ]


def train_series_model(model, train_batch, epochs, lr, agents=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        loss, acc = train_epoch_series(
            model,
            opt,
            train_batch["actions"],
            train_batch["rewards"],
            train_batch["transitions"],
            agents=agents,
        )
        history.append((loss, acc))
    return history


def make_demo_dataset(B: int, T: int, seed: int, device: str):
    actions, rewards, transitions, phases = two_step_mb_generator(
        B=B, T=T, dwell=120, beta=3.5, p_common=0.7, seed=seed, device=device
    )
    return {
        "actions": actions.long(),
        "rewards": rewards.long(),
        "transitions": transitions.long(),
        "phases": phases.long(),
    }


def split_train_test(batch: Dict[str, torch.Tensor], *, holdout: float = 0.2):
    num_sessions = batch["actions"].shape[0]
    split = max(1, int(round(num_sessions * (1 - holdout))))
    indices = torch.randperm(num_sessions)
    train_ix = indices[:split]
    test_ix = indices[split:]
    if test_ix.numel() == 0:
        test_ix = train_ix.clone()
    train = {k: v.index_select(0, train_ix) for k, v in batch.items()}
    test = {k: v.index_select(0, test_ix) for k, v in batch.items()}
    return train, test


def posterior_payload(label: str, extras: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], metrics):
    payload = {
        "label": label,
        "posterior": tensor_first_sequence(extras["gamma"]).tolist(),
    }
    if "phases" in batch:
        payload["phases"] = tensor_first_sequence(batch["phases"]).tolist()
    gating = extras.get("gating")
    if gating is not None:
        payload["gating"] = tensor_first_sequence(gating).tolist()
    best_perm = metrics.get("best_permutation")
    if best_perm is not None:
        payload["best_permutation"] = best_perm
    return payload


def main():
    parser = argparse.ArgumentParser(description="Fit HMM-MoA and TinyRNN on real behavioural data")
    parser.add_argument("--data", type=Path, help="Path to preprocessed .npz behavioural bundle")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_data"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden-moa", dest="hidden_moa", type=int, default=6)
    parser.add_argument("--hidden-rnn", dest="hidden_rnn", type=int, default=6)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1.25)
    parser.add_argument("--sticky", type=float, default=0.97)
    parser.add_argument("--holdout", type=float, default=0.2, help="Fraction of sessions reserved for evaluation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-artifacts", action="store_true", help="Persist checkpoints and posterior tensors")
    parser.add_argument(
        "--demo-synthetic",
        action="store_true",
        help="Ignore --data and run on a locally generated surrogate dataset (for CI/testing)",
    )
    parser.add_argument("--demo-B", type=int, default=4)
    parser.add_argument("--demo-T", type=int, default=200)
    parser.add_argument("--demo-seed", type=int, default=13)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_synthetic:
        print("[demo] generating surrogate dataset for pipeline smoke test")
        full_batch = make_demo_dataset(args.demo_B, args.demo_T, args.demo_seed, device="cpu")
    else:
        if args.data is None:
            raise SystemExit("--data must be provided unless --demo-synthetic is used")
        if not args.data.exists():
            raise SystemExit(f"could not find dataset file {args.data}")
        print(f"[data] loading {args.data}")
        full_batch = load_npz_dataset(args.data, device="cpu")

    train_batch, test_batch = split_train_test(full_batch, holdout=args.holdout)
    train_device = to_device(train_batch, device)
    test_device = to_device(test_batch, device)

    agent_suite = default_agent_suite()
    agents = agent_suite

    print("[hmm-moa] fitting SeriesHMMTinyMoARNN")
    hmm_moa = SeriesHMMTinyMoARNN(
        n_agents=len(agents), hidden=args.hidden_moa, K=args.K, tau=args.tau
    ).to(device)
    init_sticky(hmm_moa, stay=args.sticky, eps=1e-3)
    history_moa = train_series_model(hmm_moa, train_device, args.epochs, args.lr, agents=agents)
    history_moa_json = build_history(history_moa)
    train_metrics_moa, train_extras_moa = evaluate_series(hmm_moa, train_device, agents=agents)
    test_metrics_moa, test_extras_moa = evaluate_series(hmm_moa, test_device, agents=agents)
    dump_training_artifacts(
        args.out_dir / "hmm_moa",
        hmm_moa,
        history_moa_json,
        {"train": train_metrics_moa, "test": test_metrics_moa},
        save_checkpoint=args.save_artifacts,
    )
    write_json(
        args.out_dir / "hmm_moa" / "posterior_trace.json",
        posterior_payload("HMM-MoA", test_extras_moa, test_batch, test_metrics_moa),
    )

    print("[hmm-tinyrnn] fitting SeriesHMMTinyRNN")
    hmm_rnn = SeriesHMMTinyRNN(hidden=args.hidden_rnn, K=args.K, tau=args.tau).to(device)
    init_sticky(hmm_rnn, stay=args.sticky, eps=1e-3)
    history_rnn = train_series_model(hmm_rnn, train_device, args.epochs, args.lr, agents=None)
    history_rnn_json = build_history(history_rnn)
    train_metrics_rnn, train_extras_rnn = evaluate_series(hmm_rnn, train_device, agents=None)
    test_metrics_rnn, test_extras_rnn = evaluate_series(hmm_rnn, test_device, agents=None)
    dump_training_artifacts(
        args.out_dir / "hmm_tinyrnn",
        hmm_rnn,
        history_rnn_json,
        {"train": train_metrics_rnn, "test": test_metrics_rnn},
        save_checkpoint=args.save_artifacts,
    )
    write_json(
        args.out_dir / "hmm_tinyrnn" / "posterior_trace.json",
        posterior_payload("HMM-TinyRNN", test_extras_rnn, test_batch, test_metrics_rnn),
    )

    # Trial-history regressions (observed vs models vs individual agents)
    predictions = {
        "HMM-MoA": test_extras_moa["pi_log"].argmax(dim=-1).cpu(),
        "HMM-TinyRNN": test_extras_rnn["pi_log"].argmax(dim=-1).cpu(),
    }
    agent_predictions = agent_action_sequences(agent_suite, test_batch)
    predictions.update(agent_predictions)
    history_results = summarise_trial_history(test_batch, predictions=predictions, max_lag=5)
    write_json(
        args.out_dir / "trial_history.json",
        {
            "lags": 5,
            "series": [
                {
                    "label": result.label,
                    "reward": list(result.reward),
                    "transition": list(result.transition),
                    "interaction": list(result.interaction),
                    "common_reward": list(result.common_reward),
                    "common_omission": list(result.common_omission),
                    "rare_reward": list(result.rare_reward),
                    "rare_omission": list(result.rare_omission),
                }
                for result in history_results
            ],
        },
    )

    # Trial-history regressions (observed vs models vs individual agents)
    predictions = {
        "SeriesHMM-TinyMoA": test_extras_moa["pi_log"].argmax(dim=-1).cpu(),
        "SeriesHMM-TinyRNN": test_extras_rnn["pi_log"].argmax(dim=-1).cpu(),
    }
    agent_predictions = agent_action_sequences(agent_suite, test_batch)
    predictions.update(agent_predictions)
    history_results = summarise_trial_history(test_batch, predictions=predictions, max_lag=5)
    write_json(
        args.out_dir / "trial_history.json",
        {
            "lags": 5,
            "series": [
                {
                    "label": result.label,
                    "reward": list(result.reward),
                    "choice": list(result.choice),
                    "interaction": list(result.interaction),
                }
                for result in history_results
            ],
        },
    )

    print(
        f"[summary] HMM-MoA test acc={test_metrics_moa['accuracy']:.3f}, "
        f"HMM-TinyRNN test acc={test_metrics_rnn['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()

