# Real Data Visualizations

This document summarises each figure produced in `results/real_data/demo_fig` so that the plots can be interpreted quickly when presenting the real-data analysis.

## `real_demo_action_accuracy.svg`
Shows how accurately each evaluated model predicts the agent's actions on held-out trials. Higher accuracy indicates a closer alignment between the model's policy and the observed choices. Compare the lines to assess which agent class best matches behaviour across the session.

## `real_demo_phase_accuracy.svg`
Breaks action-prediction accuracy down by experimental phase. This highlights whether model fit changes as the task context switches (e.g., between training, probe, or reversal blocks). Diverging traces signal phase-specific strengths or weaknesses.

## `real_demo_train_accuracy.svg`
Tracks the action-prediction accuracy on the training split over optimisation epochs. A rising trend indicates the learning procedure is improving the fit to observed choices; plateaus or dips suggest convergence or overfitting.

## `real_demo_train_nll.svg`
Plots the negative log-likelihood on the training split over epochs. Lower values correspond to better fits. Inspect alongside the training-accuracy plot to ensure improvements in accuracy coincide with reductions in loss.

## `real_demo_agent_mix_serieshmm_tinymoa.svg`
Plots the responsibility each Mixture-of-Agents policy head receives inside the SeriesHMM-TinyMoA model. Curves track the weight assigned to the MF reward, MF choice, model-based, and bias experts, while the coloured ribbon along the bottom highlights which expert dominates each trial.

## `real_demo_state_posterior_serieshmm_tinymoa.svg`
Shows the posterior probabilities over latent HMM phases for the SeriesHMM-TinyMoA run. Sustained dominance of one colour indicates that the hybrid remains in a particular behavioural regime, whereas rapid colour flips correspond to phase switches.

## `real_demo_state_posterior_serieshmm_tinyrnn.svg`
Presents the posterior over hidden states for the SeriesHMM-TinyRNN model. Comparing these bands with the TinyMoA posterior reveals whether the neural head carves up the session into similar regimes or explains behaviour with different phase transitions.

## `real_demo_trial_history_observed.svg`
Plots the regression-derived stay biases from the empirical choices, decomposed into common/rare transitions crossed with reward/omission outcomes. This is the ground-truth behavioural fingerprint against which model predictions are evaluated.

## `real_demo_trial_history_agent_bias.svg`
Shows the baseline stay-bias component predicted by the Bias-only agent. Because it lacks reward or transition sensitivity, the plot should emphasise uniform biases across conditions, contrasting with the richer structure in the observed data.

## `real_demo_trial_history_agent_mf_choice.svg`
Depicts the trial-history weights learned by the Model-Free (choice-based) agent. Strong weights in rewarded conditions demonstrate habitual reinforcement, while lack of transition modulation differentiates it from the model-based profile.

## `real_demo_trial_history_agent_mf_reward.svg`
Illustrates the Model-Free (reward-value) agent's stay biases, which depend on reward history without conditioning on transition type. Compare against the observed plot to assess how well simple reward tracking captures behaviour.

## `real_demo_trial_history_agent_model_based.svg`
Shows the Model-Based agent's predicted stay biases, incorporating both transition probabilities and outcomes. Alignment with the observed fingerprints indicates the agent successfully captures planning-style adjustments after common versus rare transitions.

## `real_demo_trial_history_serieshmm_tinymoa.svg`
Summarises the stay-bias weights produced by the SeriesHMM-TinyMoA hybrid. The plot reflects how the HMM-regularised mixture balances MB and MF influences, revealing whether the hybrid reproduces the empirical common/rare asymmetries.

## `real_demo_trial_history_serieshmm_tinyrnn.svg`
Presents the SeriesHMM-TinyRNN hybrid's stay-bias predictions. It highlights how the neural dynamics, constrained by the SeriesHMM structure, account for reward and transition effects relative to the observed behaviour.

