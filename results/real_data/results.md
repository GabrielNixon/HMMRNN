# Real-data pipeline status

We introduced `series_hmm_rnn.run_real_data_pipeline` to reproduce the HMM-MoA and HMM-TinyRNN experiments on behavioural sessions released in the [MixtureAgentsModels](https://github.com/BrodyJo/MixtureAgentsModels) repository. The tool expects a converted NumPy bundle (`.npz`) containing per-session action, reward, and transition sequences; an optional `phases` array enables permutation-invariant phase accuracy reporting.

Due to network restrictions inside the execution environment we cannot download the original MATLAB files, so the runs committed in this repository use only the synthetic smoke-test mode (`--demo-synthetic`). Replace that flag with `--data path/to/converted_sessions.npz` after preparing the real dataset locally.

```bash
python -m series_hmm_rnn.run_real_data_pipeline \
  --data data/mixture_agents_sessions.npz \
  --out-dir results/real_data/mixture_agents \
  --epochs 150 \
  --device cpu
```

## Conversion workflow

1. Clone the MixtureAgentsModels repository locally where network access is permitted.
2. Convert the MATLAB structures to NumPy arrays using SciPy's `loadmat` function and save the arrays with the keys `actions`, `rewards`, `transitions`, and (optionally) `phases`. A helper skeleton is provided in `scripts/convert_mixture_agents.py`.
3. Copy the resulting `.npz` file into the `data/` directory of this project (the directory is gitignored).
4. Run the pipeline command above to generate per-model histories, metrics, and posterior traces under `results/real_data/`.

## Repository-hosted demo outputs

The committed dry-run (`--demo-synthetic`) lives in [`results/real_data/demo/`](./demo/) and mirrors the directory structure produced by a true dataset:

- [`history.json`](./demo/hmm_moa/history.json) / [`metrics.json`](./demo/hmm_moa/metrics.json) / [`posterior_trace.json`](./demo/hmm_moa/posterior_trace.json) for the HMM-MoA fit.
- [`history.json`](./demo/hmm_tinyrnn/history.json) / [`metrics.json`](./demo/hmm_tinyrnn/metrics.json) / [`posterior_trace.json`](./demo/hmm_tinyrnn/posterior_trace.json) for the HMM-TinyRNN fit.

A companion plotting pass generated SVG summaries in [`results/real_data/demo_fig/`](./demo_fig/), which you can compare side-by-side with the synthetic figures under `results/synthetic_run1/`:

- [`real_demo_train_nll.svg`](./demo_fig/real_demo_train_nll.svg) / [`real_demo_train_accuracy.svg`](./demo_fig/real_demo_train_accuracy.svg)
- [`real_demo_action_accuracy.svg`](./demo_fig/real_demo_action_accuracy.svg) / [`real_demo_phase_accuracy.svg`](./demo_fig/real_demo_phase_accuracy.svg)

### Demo metrics snapshot

| Model         | Split | NLL   | Action acc | Phase acc |
|---------------|-------|-------|------------|-----------|
| HMM-MoA       | Train | 0.920 | 0.567      | 0.538     |
| HMM-MoA       | Test  | 0.909 | 0.575      | 0.635     |
| HMM-TinyRNN   | Train | 0.656 | 0.528      | 0.935     |
| HMM-TinyRNN   | Test  | 0.667 | 0.490      | 0.935     |

These demo artefacts validate the pipeline end-to-end and provide the exact file layout to drop in the genuine real-data outputs once the converted sessions are available.
