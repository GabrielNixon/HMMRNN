# Synthetic generator walkthrough

This note expands on the synthetic benchmarking pipeline by illustrating the
latent process, the observable traces, and how the two emission models recover
the hidden phase structure.

## Latent phase dynamics

The generator alternates between two phases that favour different stage-two
rewards. Each block lasts roughly `dwell ± 10` trials before a forced switch,
producing long contiguous segments in a single latent phase. To mirror that
structure, the fitted HMM is initialised with a sticky transition matrix whose
diagonal entries are ≈0.97, corresponding to an expected dwell of about 33
steps and encouraging the posterior to preserve phase persistence.

![Sticky transition matrix](../fig/synthetic_demo_transition_matrix.svg)

## Example trajectory

The first test trajectory shows how actions, observed transitions, and rewards
co-vary with the latent phase. Blue actions favour the common transition,
orange actions favour the rare transition, and rewards are shaded dark when a
unit payoff is delivered. The latent phase row highlights the long contiguous
segments, matching the deterministic dwell structure above.

![Sequence overview](../fig/synthetic_demo_sequence_overview.svg)

## Training curves

Both the SeriesHMM-TinyMoA and SeriesHMM-TinyRNN models optimise the negative
log-likelihood (NLL) on the synthetic batches. The TinyRNN converges faster and
reaches higher action accuracy while the MoA plateaus earlier.

![Training NLL](../fig/synthetic_demo_train_nll.svg)
![Training accuracy](../fig/synthetic_demo_train_accuracy.svg)

## Posterior recovery

After 50 training epochs, the HMM posteriors show how each model tracks the
latent phase on the held-out test rollouts. The TinyRNN variant produces sharp
posterior changes that almost perfectly align with the ground-truth switches,
while the MoA head reacts more slowly and lags during certain transitions.

![SeriesHMM-TinyMoA posterior](../fig/synthetic_demo_hmm_moa_posterior.svg)
![SeriesHMM-TinyRNN posterior](../fig/synthetic_demo_hmm_tinyrnn_posterior.svg)

## Action accuracy comparison

Aggregated train/test action accuracy highlights the TinyRNN smoothing benefit:
it attains substantially higher accuracy on both splits than the MoA head under
the same sticky HMM prior.

![Accuracy summary](../fig/synthetic_demo_action_accuracy.svg)

---

*Reproduction*: regenerate the figures by running the synthetic pipeline and
plotting helper:

```bash
python -m series_hmm_rnn.run_synthetic_pipeline --epochs 50 --B 16 --T 200 \
    --out-dir outputs/figure_run --device cpu
python scripts/plot_synthetic_results.py outputs/figure_run --out-dir fig \
    --prefix synthetic_demo
```
