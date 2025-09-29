# HMMRNN

Hybrid **Hidden Markov Model + Recurrent “mixture-of-agents”** for sequential decision data.

This repo implements:

- **TinyMoA-RNN** — a compact GRU that mixes simple “agents” (MF-reward, MF-choice, MB, Bias).
- **Series-HMM-TinyMoARNN** — an HMM over *phase-specific* TinyMoA heads where **emissions are the per-phase action likelihoods**. Posteriors are computed via forward–backward and the final policy is the posterior-weighted mixture across phases.

## TL;DR results (synthetic)

## Two representative runs you can reproduce:

Run A (“decent baseline”)

Tiny -> NLL: 0.737 Acc: 0.540
Series-> NLL: 0.656 Acc: 0.626 PhaseAcc: 0.574

## Run B (same settings, different seed)

Tiny -> NLL: 0.673 Acc: 0.594 \
Series-> NLL: 0.606 Acc: 0.629 PhaseAcc: 0.515

> **Note:** Raw `PhaseAcc` can vary due to **label switching** (phase IDs are arbitrary). For a robust metric, use *permutation-invariant* PhaseAcc (see below).

---

## Repo layout

series_hmm_rnn/ \
init.py \
agents.py # MF-Reward, MF-Choice, MB, Bias \
utils.py # Q-builders, losses \
models.py # TinyMoARNN, DiscreteHMM, SeriesHMMTinyMoARNN \
data.py # two-step synthetic generator \
train.py # train/eval loops \
metrics.py # permutation-invariant phase accuracy \
scripts/ \
run_series_experiment.py \
configs/ # optional configs \
fig/ # saved figures (optional) \
checkpoints/ # optional local checkpoints \


