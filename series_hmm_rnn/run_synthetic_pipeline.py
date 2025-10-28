import argparse
import json
from pathlib import Path

import torch

from .agents import MFReward, MFChoice, BiasAgent, MBReward
from .data import two_step_mb_generator
from .metrics import phase_accuracy_permuted, align_gamma
from .models import SeriesHMMTinyMoARNN, SeriesHMMTinyRNN
from .train import train_epoch_series, eval_epoch_series


def default_agents():
    return [
        ("MFr", MFReward(alpha=0.3, decay=0.0)),
        ("MFc", MFChoice(kappa=0.2, rho=0.0)),
        ("MB", MBReward(p_common=0.7, alpha_state=0.2)),
        ("Bias", BiasAgent(0.0, 0.0)),
    ]


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


def generate_dataset(B, T, dwell, beta, p_common, seed):
    actions, rewards, transitions, states = two_step_mb_generator(
        B=B, T=T, dwell=dwell, beta=beta, p_common=p_common, seed=seed, device="cpu"
    )
    return {
        "actions": actions.long(),
        "rewards": rewards.long(),
        "transitions": transitions.long(),
        "states": states.long(),
    }


def save_tensor_dict(path, tensor_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_dict = {k: v.cpu() for k, v in tensor_dict.items()}
    torch.save(cpu_dict, path)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def maybe_cpu(tensor_or_none):
    if tensor_or_none is None:
        return None
    return tensor_or_none.cpu()


def evaluate_series(model, batch, agents=None):
    loss, acc, gk, lg = eval_epoch_series(
        model,
        batch["actions"],
        batch["rewards"],
        batch["transitions"],
        agents=agents,
    )
    gamma = torch.softmax(lg, dim=-1)
    metrics = {"nll": float(loss), "accuracy": float(acc)}
    extras = {"gamma": gamma, "gating": gk}
    if "states" in batch:
        phase_acc, perm, confusion = phase_accuracy_permuted(gamma, batch["states"])
        metrics.update(
            {
                "phase_accuracy": float(phase_acc),
                "best_permutation": list(perm),
                "confusion": confusion.cpu().tolist(),
            }
        )
        extras["aligned_gamma"] = align_gamma(gamma, perm)
    return metrics, extras


def train_series_model(model, train_batch, epochs, lr, agents=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        loss, acc = train_epoch_series(
            model,
            opt,
            train_batch["actions"],
            train_batch["rewards"],
            train_batch["transitions"],
            agents=agents,
        )
        history.append({"epoch": epoch + 1, "train_nll": float(loss), "train_accuracy": float(acc)})
    return history


def dump_training_artifacts(out_dir, model, history, eval_results):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(eval_results, indent=2))
    torch.save({"state_dict": model.state_dict()}, out_dir / "model.pt")


def main():
    parser = argparse.ArgumentParser(description="Synthetic HMM-MoA vs TinyRNN pipeline")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/synthetic_suite"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--dwell", type=int, default=120)
    parser.add_argument("--beta", type=float, default=3.5)
    parser.add_argument("--p-common", dest="p_common", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-moa", dest="hidden_moa", type=int, default=6)
    parser.add_argument("--hidden-rnn", dest="hidden_rnn", type=int, default=6)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1.25)
    parser.add_argument("--sticky", type=float, default=0.97)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] generating synthetic dataset (B={args.B}, T={args.T})")
    train_data = generate_dataset(args.B, args.T, args.dwell, args.beta, args.p_common, args.seed)
    test_data = generate_dataset(args.B, args.T, args.dwell, args.beta, args.p_common, args.seed + 1)
    data_dir = args.out_dir / "data"
    save_tensor_dict(data_dir / "train.pt", train_data)
    save_tensor_dict(data_dir / "test.pt", test_data)

    train_device = to_device(train_data, device)
    test_device = to_device(test_data, device)

    agents = default_agents()

    print("[hmm-moa] fitting SeriesHMMTinyMoARNN")
    hmm_moa = SeriesHMMTinyMoARNN(
        n_agents=len(agents), hidden=args.hidden_moa, K=args.K, tau=args.tau
    ).to(device)
    init_sticky(hmm_moa, stay=args.sticky, eps=1e-3)
    history_moa = train_series_model(hmm_moa, train_device, args.epochs, args.lr, agents=agents)
    train_metrics_moa, train_extras_moa = evaluate_series(hmm_moa, train_device, agents=agents)
    test_metrics_moa, test_extras_moa = evaluate_series(hmm_moa, test_device, agents=agents)
    results_moa = {"train": train_metrics_moa, "test": test_metrics_moa}
    dump_training_artifacts(args.out_dir / "hmm_moa", hmm_moa, history_moa, results_moa)
    torch.save(
        {
            "gamma": test_extras_moa["gamma"].cpu(),
            "aligned_gamma": maybe_cpu(test_extras_moa.get("aligned_gamma")),
            "gating": maybe_cpu(test_extras_moa.get("gating")),
            "states": test_device["states"].cpu(),
            "best_permutation": test_metrics_moa.get("best_permutation"),
        },
        args.out_dir / "hmm_moa" / "posterior_test.pt",
    )

    print(
        f"[hmm-moa] test NLL={test_metrics_moa['nll']:.3f}  "
        f"Acc={test_metrics_moa['accuracy']:.3f}  "
        f"PhaseAcc={test_metrics_moa.get('phase_accuracy', float('nan')):.3f}"
    )

    print("[hmm-tinyrnn] fitting SeriesHMMTinyRNN")
    hmm_rnn = SeriesHMMTinyRNN(hidden=args.hidden_rnn, K=args.K, tau=args.tau).to(device)
    init_sticky(hmm_rnn, stay=args.sticky, eps=1e-3)
    history_rnn = train_series_model(hmm_rnn, train_device, args.epochs, args.lr, agents=None)
    train_metrics_rnn, train_extras_rnn = evaluate_series(hmm_rnn, train_device, agents=None)
    test_metrics_rnn, test_extras_rnn = evaluate_series(hmm_rnn, test_device, agents=None)
    results_rnn = {"train": train_metrics_rnn, "test": test_metrics_rnn}
    dump_training_artifacts(args.out_dir / "hmm_tinyrnn", hmm_rnn, history_rnn, results_rnn)
    torch.save(
        {
            "gamma": test_extras_rnn["gamma"].cpu(),
            "aligned_gamma": maybe_cpu(test_extras_rnn.get("aligned_gamma")),
            "states": test_device["states"].cpu(),
            "best_permutation": test_metrics_rnn.get("best_permutation"),
        },
        args.out_dir / "hmm_tinyrnn" / "posterior_test.pt",
    )

    print(
        f"[hmm-tinyrnn] test NLL={test_metrics_rnn['nll']:.3f}  "
        f"Acc={test_metrics_rnn['accuracy']:.3f}  "
        f"PhaseAcc={test_metrics_rnn.get('phase_accuracy', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
