# Synthetic pipeline visualisations

These figures were generated from `results/synthetic_run1` using `scripts/plot_synthetic_results.py`. They compare the HMM-MoA and HMM-TinyRNN variants trained on the long-dwell synthetic dataset.

![Training negative log-likelihood trends](../fig/synthetic_run1_train_nll.svg)

*Both models converge rapidly, with the TinyRNN maintaining a consistently lower training NLL than the MoA head.*

![Training action accuracy](../fig/synthetic_run1_train_accuracy.svg)

*The TinyRNN achieves noticeably higher action accuracy throughout training, highlighting the benefit of the smooth recurrent emission.*

![Action accuracy by split](../fig/synthetic_run1_action_accuracy.svg)

*Bar chart of train/test action accuracy. The TinyRNN generalises better to held-out trajectories.*

![Phase accuracy by split](../fig/synthetic_run1_phase_accuracy.svg)

*Bar chart of train/test phase accuracy. The TinyRNN nearly recovers the latent phases, while the MoA baseline lags behind.*

![Agent mixture weights for the HMM-MoA head](../fig/synthetic_run1_hmm_moa_agent_mixture.svg)

*Per-trial mixture weights for the Model-free value, Model-free choice, Model-based, and Bias agents after aligning the latent
phases. The new labels make it easy to read off how strongly each component contributes at every point in the trajectory.*
