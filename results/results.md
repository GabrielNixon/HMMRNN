# Synthetic HMM-MoA vs HMM-TinyRNN Results

This run executes `series_hmm_rnn.run_synthetic_pipeline` with the following key arguments:

- Epochs: 50
- Batch size (trajectories): 16
- Sequence length: 200
- Hidden units: 6 for both emissions
- Temperature (tau): 1.25
- Stickiness: 0.97
- Device: CPU

Textual logs for this run live under [`results/synthetic_run1/`](synthetic_run1/).

## Evaluation metrics

| Model | Split | NLL | Action accuracy | Phase accuracy |
| --- | --- | --- | --- | --- |
| HMM-MoA | Train | 0.804 | 0.548 | 0.538 |
| HMM-MoA | Test | 0.817 | 0.553 | 0.528 |
| HMM-TinyRNN | Train | 0.586 | 0.735 | 0.993 |
| HMM-TinyRNN | Test | 0.583 | 0.746 | 0.988 |

Confusion matrices for the phase alignment can be found in [`hmm_moa/metrics.json`](synthetic_run1/hmm_moa/metrics.json) and [`hmm_tinyrnn/metrics.json`](synthetic_run1/hmm_tinyrnn/metrics.json).

## Reproducing

To regenerate these numbers without producing binary artefacts, run

```
python -m series_hmm_rnn.run_synthetic_pipeline --epochs 50 --B 16 --T 200 --out-dir results/synthetic_run1 --device cpu
```

If you do need checkpoints, posterior dumps, or the raw simulated datasets, append `--save-artifacts` to the command above. Binary blobs will then be written locally but they remain `.gitignore`d so they are not accidentally committed.
