# Real-data pipeline status

We introduced `series_hmm_rnn.run_real_data_pipeline` to reproduce the HMM-MoA and HMM-TinyRNN experiments on behavioural sessions released in the [MixtureAgentsModels](https://github.com/BrodyJo/MixtureAgentsModels) repository. The tool expects a converted NumPy bundle (``.npz``) containing per-session action, reward, and transition sequences; an optional ``phases`` array enables permutation-invariant phase accuracy reporting.

Due to network restrictions inside the execution environment we cannot download the original MATLAB files, so the run documented here uses only the synthetic smoke-test mode (``--demo-synthetic``). Replace that flag with ``--data path/to/converted_sessions.npz`` after preparing the real dataset locally.

```bash
python -m series_hmm_rnn.run_real_data_pipeline \
  --data data/mixture_agents_sessions.npz \
  --out-dir results/real_data/mixture_agents \
  --epochs 150 \
  --device cpu
```

## Conversion workflow

1. Clone the MixtureAgentsModels repository locally where network access is permitted.
2. Convert the MATLAB structures to NumPy arrays using SciPy's ``loadmat`` function and save the arrays with the keys ``actions``, ``rewards``, ``transitions``, and (optionally) ``phases``. A helper skeleton is provided in `scripts/convert_mixture_agents.py`.
3. Copy the resulting ``.npz`` file into the ``data/`` directory of this project (the directory is gitignored).
4. Run the pipeline command above to generate per-model histories, metrics, and posterior traces under ``results/real_data/``.

## Current smoke-test outputs

The synthetic dry-run (``--demo-synthetic``) confirms that the pipeline executes end-to-end without the real dataset. Once the external data are accessible, rerun the command without the demo flag to populate this section with genuine metrics.
