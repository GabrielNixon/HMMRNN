# Main branch results overview

This document collects the synthetic benchmark assets that were previously scattered across development branches and places them
alongside the `main` lineage of the repository. It summarises the training runs, links to raw JSON logs, and points to the text-
only visualisations that can be regenerated directly from source.

## Experiments in scope

- **HMM-MoA** (mixture-of-agents emissions) trained on the long-dwell two-step generator.
- **HMM-TinyRNN** (smooth recurrent emissions) trained on the same dataset with shared hyperparameters.
- A real-data pipeline (`series_hmm_rnn/run_real_data_pipeline.py`) that mirrors the synthetic workflow once MixtureAgentsModels
  sessions are converted into the expected NumPy bundle.
- Supporting TinyMoA-only fits for debugging, and the plotting scripts that convert JSON logs into SVG figures.

The pipeline that orchestrates the dataset synthesis, training, and logging lives in
`series_hmm_rnn/run_synthetic_pipeline.py`. 【F:series_hmm_rnn/run_synthetic_pipeline.py†L1-L222】

## Key metrics

| Model | Split | NLL | Action accuracy | Phase accuracy |
| --- | --- | --- | --- | --- |
| HMM-MoA | Train | 0.804 | 0.548 | 0.538 |
| HMM-MoA | Test | 0.817 | 0.553 | 0.528 |
| HMM-TinyRNN | Train | 0.586 | 0.735 | 0.993 |
| HMM-TinyRNN | Test | 0.583 | 0.746 | 0.988 |

These numbers are consolidated from [`results/results.md`](results/results.md), which in turn is derived from the JSON metrics
dumped by the pipeline. 【F:results/results.md†L1-L40】

## Raw artefacts

- Training histories: [`results/synthetic_run1/hmm_moa/history.json`](results/synthetic_run1/hmm_moa/history.json) and
  [`results/synthetic_run1/hmm_tinyrnn/history.json`](results/synthetic_run1/hmm_tinyrnn/history.json).
- Evaluation metrics (including permutation-aligned confusion matrices):
  [`results/synthetic_run1/hmm_moa/metrics.json`](results/synthetic_run1/hmm_moa/metrics.json) and
  [`results/synthetic_run1/hmm_tinyrnn/metrics.json`](results/synthetic_run1/hmm_tinyrnn/metrics.json).
- Visual summary: [`results/visualizations.md`](results/visualizations.md) references the SVG charts produced by
  `scripts/plot_synthetic_results.py`. 【F:results/visualizations.md†L1-L18】【F:scripts/plot_synthetic_results.py†L1-L333】

### Real-data demo companion

- Summary + figure links: [`results/real_data/README.md`](results/real_data/README.md).
- JSON logs: [`results/real_data/demo/`](results/real_data/demo/) stores histories, metrics, posterior traces, and a sample
  trajectory.
- SVG plots (NLL, accuracy, posterior traces, trial-history regressions):
  [`results/real_data/demo_fig/`](results/real_data/demo_fig/).

## How to reproduce

1. **Run the synthetic pipeline** to regenerate the JSON logs without producing binary artefacts:

   ```bash
   python -m series_hmm_rnn.run_synthetic_pipeline --epochs 50 --B 16 --T 200 --out-dir results/synthetic_run1 --device cpu
   ```

2. **Optionally enable artefact saving** for checkpoints, posterior dumps, and raw datasets by appending `--save-artifacts`. The
   `.gitignore` keeps these binaries out of version control. 【F:results/results.md†L31-L40】

3. **Regenerate SVG figures** from the JSON logs if desired:

   ```bash
 python scripts/plot_synthetic_results.py results/synthetic_run1 --out-dir fig --prefix synthetic_run1
  ```

4. **Process MixtureAgentsModels sessions** (optional real-data workflow):

   ```bash
   python -m series_hmm_rnn.run_real_data_pipeline \
     --data data/mixture_agents_sessions.npz \
     --out-dir results/real_data/mixture_agents \
     --epochs 150 \
     --device cpu
   ```

   Use `--demo-synthetic` for a quick smoke test when the `.mat` files are not
   accessible.

5. **Inspect TinyMoA-specific diagnostics** using the standalone CLI:

   ```bash
   python -m series_hmm_rnn.run_tiny_fit --epochs 10 --B 8 --T 120 --trace-out '' --device cpu
   ```

   Use `--plot-out` or `--history-out` when you need richer per-epoch traces. 【F:series_hmm_rnn/run_tiny_fit.py†L1-L354】

With this collation in place, the main branch exposes the entire HMM-MoA vs TinyRNN workflow — models, training scripts, metrics,
and figures — through text files that remain compatible with lightweight review tooling.
