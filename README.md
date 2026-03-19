# HMMRNN

A compact research codebase for modelling sequential decision-making with a **Hidden Markov Model (HMM)** and **recurrent neural networks (RNNs)**.

This repository compares two related model families on a two-step behavioural task:

- **SeriesHMM-TinyMoA**: an HMM with per-phase **mixtures of interpretable agents**.
- **SeriesHMM-TinyRNN**: an HMM with per-phase **learned recurrent policy heads**.

The project is set up so you can:

1. generate synthetic sequences with known latent phases,
2. train both model variants under matched settings,
3. evaluate action prediction and latent-state recovery,
4. run the same workflow on external behavioural datasets, and
5. inspect training curves, posteriors, and trial-history summaries.

---

## Research question

The main question in this repository is:

> **Can a compact HMM-based sequential model recover latent behavioural modes and predict actions in decision tasks, and how does an interpretable mixture-of-agents emission compare with a more flexible recurrent emission?**

In practice, the code studies whether behaviour is better captured by:

- a structured **mixture of canonical decision agents** such as model-free, model-based, and bias components, or
- a smoother but less hand-structured **TinyRNN-style emission model**.

The HMM provides the latent phase structure, while the emission model determines how actions are produced within each phase.

---

## What we implemented

### 1. Synthetic behavioural environment

The synthetic generator creates long sequences from a two-step decision task with:

- two latent phases,
- sticky dwell periods,
- common and rare transitions,
- binary rewards, and
- ground-truth latent labels for evaluation.

This makes it possible to test whether the models recover hidden behavioural regimes, not just surface-level action frequencies.

### 2. TinyMoA baseline

`TinyMoARNN` is a compact GRU-based recurrent model that mixes a small library of decision agents into action values over time.

### 3. SeriesHMM-TinyMoA

`SeriesHMMTinyMoARNN` adds an HMM over latent phases and assigns each phase its own gating over the agent library. This keeps the model relatively interpretable because each phase can emphasize different agents.

### 4. SeriesHMM-TinyRNN

`SeriesHMMTinyRNN` keeps the same HMM backbone but replaces the hand-structured agent mixture with learned recurrent per-phase action heads.

### 5. Evaluation and visualization

The repository includes utilities for:

- negative log-likelihood,
- action accuracy,
- permutation-invariant phase accuracy,
- posterior export,
- trial-history regressions, and
- SVG plots for qualitative inspection.

---

## Repository structure

### Top-level files

- `README.md` — project overview and usage guide.
- `PROCESS.md` — design rationale for the synthetic benchmark.
- `RESULTS.md` — high-level summary of benchmark outputs.
- `TROUBLESHOOTING.md` — notes for common runtime issues.
- `pyproject.toml` — package metadata and dependencies.

### Core package: `series_hmm_rnn/`

- `__init__.py`
  Exposes the main public API for models, agents, utilities, training, and metrics.

- `agents.py`
  Defines the canonical agent library:
  - model-free reward learner,
  - model-free choice perseveration learner,
  - model-based reward learner,
  - bias agent.

- `data.py`
  Synthetic two-step task generator with latent phases, transition structure, and rewards.

- `models.py`
  Defines:
  - `TinyMoARNN`,
  - `DiscreteHMM`,
  - `SeriesHMMTinyMoARNN`,
  - `SeriesHMMTinyRNN`.

- `train.py`
  Shared training and evaluation loops for TinyMoA and SeriesHMM models.

- `utils.py`
  Input building, agent Q-value stacking, and NLL loss helpers.

- `metrics.py`
  Phase alignment and permutation-invariant latent-state accuracy.

- `trial_history.py`
  Trial-history regression utilities plus agent/model action sequence summaries.

- `run_synthetic_pipeline.py`
  End-to-end synthetic experiment runner. Generates data, trains both models, evaluates them, and writes outputs.

- `run_real_data_pipeline.py`
  Runs the same modeling pipeline on external behavioural sessions stored as `.npz` bundles.

- `run_tiny_fit.py`
  Lightweight TinyMoA-only training script for debugging and quick tests.

- `run_series_experiment.py`
  Older baseline script kept for prototyping/reference.

### Scripts: `scripts/`

- `plot_synthetic_results.py`
  Converts JSON outputs into SVG figures.

- `plot_state_posterior.py`
  Visualizes latent-state posterior trajectories.

- `plot_projected_agent_mix.py`
  Plots projected agent-mixture summaries.

- `convert_mixture_agents.py`
  Skeleton helper for converting external behavioural data into the expected `.npz` format.

- `train_tiny.py`
  Additional utility script for TinyMoA experiments.

### Results and artifacts

- `results/synthetic_run1/` — saved synthetic benchmark outputs.
- `results/real_data/` — real-data or demo-pipeline outputs.
- `fig/` — tracked SVG and image visualizations.
- `checkpoints/` — saved model checkpoints from earlier runs.
- `configs/` — configuration files for small experiments.

---

## How the models work

### Agent library

The interpretable emission model uses four canonical agent types:

1. **MF Reward** — updates values from experienced rewards.
2. **MF Choice** — captures choice repetition/perseveration.
3. **Model-based** — uses transition structure to compute action values.
4. **Bias** — captures static action preference.

### Shared sequential input

Across models, the recurrent encoder consumes a per-trial history of:

- action,
- reward,
- transition.

### HMM layer

The HMM models latent behavioural phases through:

- initial state probabilities,
- transition probabilities,
- forward-backward inference over time.

### Emission comparison

- **SeriesHMM-TinyMoA** uses phase-specific mixtures over the agent library.
- **SeriesHMM-TinyRNN** uses phase-specific learned logits from the recurrent hidden state.

This makes the comparison meaningful: both models share the same latent-state backbone, while differing mainly in how each latent state emits action probabilities.

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.2+
- PyYAML 6.0+

Install the core dependencies in your environment:

```bash
pip install torch pyyaml
```

If you plan to use external `.npz` datasets, you will also want:

```bash
pip install numpy
```

---

## Quick start

### Run the synthetic benchmark

```bash
python -m series_hmm_rnn.run_synthetic_pipeline \
  --epochs 50 \
  --B 16 \
  --T 200 \
  --out-dir results/synthetic_run1 \
  --device cpu
```

This will:

1. generate synthetic train/test trajectories,
2. fit `SeriesHMM-TinyMoA`,
3. fit `SeriesHMM-TinyRNN`,
4. export histories, metrics, and posterior traces.

### Plot the outputs

```bash
python scripts/plot_synthetic_results.py \
  results/synthetic_run1 \
  --out-dir fig \
  --prefix synthetic_run1
```

### Run the TinyMoA-only debug path

```bash
python -m series_hmm_rnn.run_tiny_fit \
  --epochs 10 \
  --B 8 \
  --T 120 \
  --trace-out '' \
  --device cpu
```

### Run the real-data pipeline

```bash
python -m series_hmm_rnn.run_real_data_pipeline \
  --data path/to/sessions.npz \
  --out-dir results/real_data/my_run \
  --epochs 150 \
  --device cpu
```

If you only want to smoke-test the pipeline without external data:

```bash
python -m series_hmm_rnn.run_real_data_pipeline \
  --demo-synthetic \
  --out-dir results/real_data/demo \
  --epochs 150 \
  --device cpu
```

---

## Input data format for real datasets

The real-data pipeline expects a `.npz` file containing rank-2 arrays of shape:

```text
[num_sessions, T]
```

Required keys:

- `actions`
- `rewards`
- `transitions`

Optional key:

- `phases`

If `phases` is provided, the evaluation will also report permutation-invariant phase accuracy.

---

## Main configuration knobs

You can change the experiments by editing command-line arguments.

### Data and sequence settings

- `--B` — number of sequences / sessions.
- `--T` — sequence length.
- `--dwell` — average latent dwell time in synthetic generation.
- `--beta` — policy sharpness used in synthetic data generation.
- `--holdout` — evaluation split fraction for real-data runs.

### Model settings

- `--K` — number of latent HMM states.
- `--hidden-moa` or `--hidden_series` — hidden size for the MoA-based model.
- `--hidden-rnn` — hidden size for the TinyRNN emission model.
- `--tau` — scaling applied to emission probabilities before HMM inference.
- `--sticky` — stickiness used to initialize the HMM transition matrix.

### Optimization settings

- `--epochs` — training epochs.
- `--lr` — learning rate.
- `--seed` or `--demo-seed` — random seed.
- `--device` — `cpu` or `cuda`.

### Artifact control

- `--save-artifacts` — save checkpoints and additional outputs.

If you want to experiment with different behavioral regimes, the most important places to start are:

1. `series_hmm_rnn/data.py` for synthetic task generation,
2. `series_hmm_rnn/agents.py` for the canonical agent definitions,
3. `series_hmm_rnn/models.py` for model structure,
4. `series_hmm_rnn/run_synthetic_pipeline.py` for experiment settings.

---

## Results

### Synthetic benchmark (`results/synthetic_run1`)

Default benchmark summary:

| Model | Split | NLL | Action accuracy | Phase accuracy |
| --- | --- | --- | --- | --- |
| SeriesHMM-TinyMoA | Train | 0.804 | 0.548 | 0.538 |
| SeriesHMM-TinyMoA | Test | 0.817 | 0.553 | 0.528 |
| SeriesHMM-TinyRNN | Train | 0.586 | 0.735 | 0.993 |
| SeriesHMM-TinyRNN | Test | 0.583 | 0.746 | 0.988 |

### What these results mean

The current synthetic benchmark suggests that:

- **SeriesHMM-TinyRNN predicts actions much better** than SeriesHMM-TinyMoA on this generator.
- **SeriesHMM-TinyRNN also recovers latent phases far more accurately** under the tracked benchmark settings.
- **SeriesHMM-TinyMoA remains useful when interpretability matters**, because its phase-specific policies are built from explicit agent mixtures rather than fully learned recurrent heads.

So the present codebase supports a familiar tradeoff:

- **TinyRNN emission** gives stronger empirical fit on the synthetic benchmark.
- **TinyMoA emission** gives stronger interpretability because its decisions can be decomposed into canonical agent contributions.

### Real-data pipeline status

The repository also includes a real-data workflow, but the tracked documentation currently reflects a synthetic smoke test rather than a full external behavioural-data benchmark. Once a converted `.npz` dataset is available locally, the same pipeline can be rerun to generate comparable metrics and figures.

---

## Outputs you should expect

After a typical run, you will usually see:

- `history.json` — per-epoch training metrics,
- `metrics.json` — train/test performance summaries,
- `posterior_trace.json` — latent posterior trajectories,
- `trial_history.json` — stay/reward/choice regression summaries,
- SVG plots in `fig/` or another chosen output directory.

---

## Suggested workflow for extending the project

If you are modifying or extending this repository, a good path is:

1. **Change the generator** in `data.py` if you want a new task structure.
2. **Add or revise agents** in `agents.py` if you want a different interpretable basis.
3. **Modify the emissions** in `models.py` if you want a new hybrid architecture.
4. **Run `run_synthetic_pipeline.py`** to benchmark the change under controlled conditions.
5. **Use the plotting scripts** to inspect whether gains are quantitative only or also interpretable.

---

## Limitations

- The synthetic generator is intentionally simple and does not cover all properties of real behavioural data.
- The real-data path depends on external preprocessing into `.npz` format.
- The tracked results show one benchmark configuration, not a full hyperparameter sweep.
- Interpretability and predictive performance are not identical goals, so model choice depends on the scientific question.

---

## Summary

This repository is a compact framework for comparing **interpretable agent-mixture HMM emissions** against **learned recurrent HMM emissions** on sequential decision tasks.

If you are here to understand the project quickly:

- start with `series_hmm_rnn/run_synthetic_pipeline.py`,
- inspect `series_hmm_rnn/models.py`,
- review the outputs under `results/`, and
- use the scripts under `scripts/` to visualize what the models learned.
