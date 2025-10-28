"""Utilities for computing trial-history regressions on two-step data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .agents import BiasAgent, MBReward, MFChoice, MFReward


@dataclass
class TrialHistoryResult:
    label: str
    reward: Sequence[float]
    choice: Sequence[float]
    interaction: Sequence[float]


def _logistic_fit(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fit a logistic regression with an intercept using LBFGS."""

    if targets.numel() == 0 or torch.isclose(targets.float().var(unbiased=False), torch.tensor(0.0)):
        return torch.zeros(features.shape[1], dtype=torch.float32)

    X = torch.cat([torch.ones(features.size(0), 1), features], dim=1)
    y = targets.float()

    params = torch.zeros(X.size(1), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():  # type: ignore[no-untyped-def]
        optimizer.zero_grad()
        logits = X.matmul(params)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return params[1:].detach().clone()


def _build_design_matrix(
    actions: torch.Tensor, rewards: torch.Tensor, *, max_lag: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct design matrices for reward/choice/interaction regressions."""

    actions = actions.long()
    rewards = rewards.long()
    B, T = actions.shape

    stay_targets: List[float] = []
    reward_rows: List[List[float]] = []
    choice_rows: List[List[float]] = []
    interaction_rows: List[List[float]] = []

    for b in range(B):
        acts = actions[b]
        rwd = rewards[b]
        for t in range(max_lag, T):
            stay = float(acts[t] == acts[t - 1])
            stay_targets.append(stay)
            reward_feats: List[float] = []
            choice_feats: List[float] = []
            for lag in range(1, max_lag + 1):
                reward_feats.append(float(rwd[t - lag] * 2 - 1))
                choice_feats.append(float(acts[t - lag] * 2 - 1))
            reward_rows.append(reward_feats)
            choice_rows.append(choice_feats)
            interaction_rows.append([r * c for r, c in zip(reward_feats, choice_feats)])

    if not stay_targets:
        zeros = torch.zeros((0, max_lag), dtype=torch.float32)
        return zeros, zeros, zeros, torch.zeros(0, dtype=torch.float32)

    reward_mat = torch.tensor(reward_rows, dtype=torch.float32)
    choice_mat = torch.tensor(choice_rows, dtype=torch.float32)
    inter_mat = torch.tensor(interaction_rows, dtype=torch.float32)
    targets = torch.tensor(stay_targets, dtype=torch.float32)
    return reward_mat, choice_mat, inter_mat, targets


def trial_history_coefficients(
    actions: torch.Tensor,
    rewards: torch.Tensor,
    *,
    max_lag: int = 5,
) -> Mapping[str, Sequence[float]]:
    """Return reward/choice/interaction coefficients for the provided actions."""

    reward_mat, choice_mat, inter_mat, targets = _build_design_matrix(actions, rewards, max_lag=max_lag)
    coeffs = {
        "reward": _logistic_fit(reward_mat, targets).tolist(),
        "choice": _logistic_fit(choice_mat, targets).tolist(),
        "interaction": _logistic_fit(inter_mat, targets).tolist(),
    }
    return coeffs


def agent_action_sequences(
    agents: Iterable[Tuple[str, object]],
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Generate per-agent argmax policies conditioned on the observed history."""

    actions = batch["actions"]
    rewards = batch["rewards"]
    transitions = batch.get("transitions")
    T = actions.size(1)
    sequences: Dict[str, torch.Tensor] = {}

    for label, agent in agents:
        try:
            q_values = agent.forward(actions, rewards, T)  # type: ignore[attr-defined]
        except TypeError:
            try:
                q_values = agent.forward(actions, T)  # type: ignore[attr-defined]
            except TypeError:
                try:
                    q_values = agent.forward(actions, rewards, transitions, T)  # type: ignore[attr-defined]
                except TypeError:
                    q_values = agent.forward(actions.size(0), T, actions.device)  # type: ignore[attr-defined]
        sequences[f"Agent: {label}"] = q_values.argmax(dim=-1)

    return sequences


def summarise_trial_history(
    batch: Mapping[str, torch.Tensor],
    *,
    predictions: Mapping[str, torch.Tensor],
    max_lag: int = 5,
) -> List[TrialHistoryResult]:
    """Compute trial-history regressions for ground truth and supplied predictions."""

    rewards = batch["rewards"].detach().cpu().long()
    sequences: List[Tuple[str, torch.Tensor]] = [("Observed", batch["actions"].detach().cpu().long())]
    for label, seq in predictions.items():
        sequences.append((label, seq.detach().cpu().long()))

    results: List[TrialHistoryResult] = []
    for label, actions in sequences:
        coeffs = trial_history_coefficients(actions, rewards, max_lag=max_lag)
        results.append(
            TrialHistoryResult(
                label=label,
                reward=coeffs["reward"],
                choice=coeffs["choice"],
                interaction=coeffs["interaction"],
            )
        )
    return results


def default_agent_suite() -> List[Tuple[str, object]]:
    """Instantiate the canonical four-agent library used across the repo."""

    return [
        ("MF Reward", MFReward(alpha=0.3, decay=0.0)),
        ("MF Choice", MFChoice(kappa=0.2, rho=0.0)),
        ("Model-based", MBReward(p_common=0.7, alpha_state=0.2)),
        ("Bias", BiasAgent(0.0, 0.0)),
    ]

