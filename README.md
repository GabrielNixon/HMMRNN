# HMMRNN

Hybrid **Hidden Markov Model + recurrent mixture-of-agents** tools for analysing sequential decision data, following the HMM-MoA and Tiny RNN literature.

## What lives in this branch

- **TinyMoARNN** — a compact GRU that mixes simple “agents” (model-free reward/choice, model-based, bias) to produce action logits. 【F:series_hmm_rnn/models.py†L26-L178】
- **SeriesHMMTinyMoARNN** — an HMM whose emissions are the per-phase TinyMoA heads; phase posteriors are inferred with forward–backward and used to weight the agents. 【F:series_hmm_rnn/models.py†L181-L356】
- **SeriesHMMTinyRNN** — swaps the MoA emissions for phase-specific GRU heads to obtain a smooth recurrent controller. 【F:series_hmm_rnn/models.py†L359-L511】
- **Synthetic pipeline** (`series_hmm_rnn/run_synthetic_pipeline.py`) — generates long-dwell two-step trajectories, trains both models, and exports histories/metrics ready for documentation or plotting. 【F:series_hmm_rnn/run_synthetic_pipeline.py†L1-L222】
- **Plotting helper** (`scripts/plot_synthetic_results.py`) — renders lightweight SVG training curves and accuracy comparisons directly from the JSON logs. 【F:scripts/plot_synthetic_results.py†L1-L333】

## Reproduced synthetic benchmark

The repository includes a complete, text-only reproduction of the long-dwell two-step benchmark used throughout development. Aggregate metrics, confusion matrices, and figure links are gathered in [`RESULTS.md`](RESULTS.md). 【F:RESULTS.md†L1-L96】

Headline numbers from the default run (`results/synthetic_run1`) are:

| Model | Split | NLL | Action acc. | Phase acc. |
| --- | --- | --- | --- | --- |
| HMM-MoA | Train | 0.804 | 0.548 | 0.538 |
| HMM-MoA | Test | 0.817 | 0.553 | 0.528 |
| HMM-TinyRNN | Train | 0.586 | 0.735 | 0.993 |
| HMM-TinyRNN | Test | 0.583 | 0.746 | 0.988 |

Regenerate them with:

```bash
python -m series_hmm_rnn.run_synthetic_pipeline --epochs 50 --B 16 --T 200 --out-dir results/synthetic_run1 --device cpu
```

Add `--save-artifacts` if you also need checkpoints, posteriors, or datasets (kept out of git by default). 【F:results/results.md†L1-L40】

## Real-data demo artefacts

A demo run of the real-data pipeline (executed with the synthetic stand-in shipped in this
repository) lives in [`results/real_data/`](results/real_data/). The summary table and
links to the committed SVG figures — including the per-trial agent mixture traces — are
available in [`results/real_data/README.md`](results/real_data/README.md), making it easy to compare
against the synthetic benchmark.

## Quickstart

### End-to-end synthetic pipeline

```bash
python -m series_hmm_rnn.run_synthetic_pipeline --epochs 10 --B 8 --T 120 --out-dir outputs/demo --device cpu
```

This command synthesises data, trains the MoA and TinyRNN heads, and stores JSON logs plus optional figures. 【F:series_hmm_rnn/run_synthetic_pipeline.py†L15-L222】

### TinyMoA-only experiments

```bash
python -m series_hmm_rnn.run_tiny_fit --epochs 10 --B 8 --T 120 --trace-out '' --device cpu
```

Use `--plot-out` and `--history-out` to export per-epoch diagnostics for bespoke analyses. 【F:series_hmm_rnn/run_tiny_fit.py†L1-L354】

### Real-data pipeline

Prepare the MixtureAgentsModels behavioural exports with SciPy (see
`scripts/convert_mixture_agents.py`) and then run:

```bash
python -m series_hmm_rnn.run_real_data_pipeline \
  --data data/mixture_agents_sessions.npz \
  --out-dir results/real_data/mixture_agents \
  --epochs 150 \
  --device cpu
```

Add `--demo-synthetic` when you just need a smoke test without the external
dataset.  The command mirrors the synthetic workflow and writes JSON histories,
metrics, and posterior traces to the requested output directory.

### Plotting existing runs

```bash
python scripts/plot_synthetic_results.py results/synthetic_run1 --out-dir fig --prefix synthetic_run1
```

The script emits SVG summaries suitable for git-friendly visual inspection. 【F:scripts/plot_synthetic_results.py†L1-L333】

## Repository layout

```
series_hmm_rnn/
  __init__.py
  agents.py            # MF-Reward, MF-Choice, MB, Bias agents
  data.py              # two-step synthetic generator
  models.py            # TinyMoARNN, SeriesHMMTinyMoARNN, SeriesHMMTinyRNN
  train.py             # shared train/eval loops
  run_synthetic_pipeline.py
  run_tiny_fit.py
results/
  synthetic_run1/      # default benchmark artefacts (JSON only)
  results.md           # metrics table for the benchmark
  visualizations.md    # SVG figure references
scripts/
  plot_synthetic_results.py
fig/                   # generated SVG charts (text-based)
configs/               # optional experiment configs
```

The older `run_series_experiment.py` baseline used during early prototyping is still available under `series_hmm_rnn/`. 【F:series_hmm_rnn/run_series_experiment.py†L1-L208】
