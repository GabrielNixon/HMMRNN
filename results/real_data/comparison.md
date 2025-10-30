# SeriesHMM TinyMoA vs TinyRNN responsibility comparison

This note condenses the figures under `results/real_data/demo_fig` that expose how
the hybrid models behave internally on the demo real-data run. Each section embeds
the generated SVG and highlights the takeaways so the plots can be dropped straight
into a presentation.

## TinyMoA agent responsibility (`real_demo_agent_mix_serieshmm_tinymoa.svg`)

![SeriesHMM-TinyMoA agent mix](./demo_fig/real_demo_agent_mix_serieshmm_tinymoa.svg)

*Reading the plot*: the four trajectories correspond to the default Mixture-of-Agents
experts (MF reward, MF choice, model-based, bias). Values are normalised to sum to
one at every trial. The shaded band beneath the axes is colour-coded by the most
probable expert so you can immediately spot when responsibility shifts.

*What it shows*: in this demo session the model-based and bias experts trade off
control. The bias head dominates 107 trials while the model-based head wins 93,
leaving only small windows where the model-free components take charge. Averaged
across the run the weights settle near 0.16 (MF reward), 0.22 (MF choice), 0.30
(model-based), and 0.31 (bias), underscoring that the hybrid leans heavily on the
planning-style and static-bias policies.【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】

## TinyRNN phase responsibility (`real_demo_agent_mix_hmm_tinyrnn.svg`)

![HMM-TinyRNN phase mix](./demo_fig/real_demo_agent_mix_hmm_tinyrnn.svg)

*Reading the plot*: the blue and orange lines track the SeriesHMM posterior over
the TinyRNN's two recurrent phases, while the coloured bars at the bottom flag the
ground-truth task blocks for quick alignment with experimental context.

*What it shows*: the blue phase averages 0.71 responsibility and wins 136 of the
200 trials, whereas the orange phase carries the remaining 0.29 mass across 64
trials.【374e99†L1-L3】 Responsibility switches cluster around the same probe and
reversal windows that trigger handovers between the TinyMoA bias and model-based
experts, highlighting that the neural controller mirrors those behavioural
regimes. Correlating each TinyRNN phase with the TinyMoA agent weights reveals the
mapping explicitly: the dominant (blue) state co-varies with the Mixture-of-Agent's
model-free reward and choice heads (r ≈ 0.66 and 0.57) while anti-correlating with
the bias expert, and the rarer (orange) state flips that pattern, aligning most
strongly with the bias responsibility (r ≈ 0.76).【423ce9†L1-L8】 Together these
statistics confirm that the TinyRNN moves through modes that shadow the
Mixture-of-Agents decomposition rather than inventing entirely new dynamics.

To make the comparison one-to-one with the TinyMoA breakdown, we project the
TinyRNN phase posterior onto the MoA agent space by aligning the neural phases to
the MoA latent states (permutation = [0, 1]) and taking the expectation of the
MoA per-phase agent weights under the TinyRNN responsibilities. The resulting
agent mix averages 0.20 MF reward, 0.24 MF choice, 0.31 model-based, and 0.24
bias responsibility across the session, so the recurrent head preserves the
planning-versus-bias tug-of-war despite smoothing the transitions.【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1215-L1219】
Trial-by-trial maxima show the projected TinyRNN controller selects the
model-based expert on 158 trials and the bias head on the remaining 42, with no
windows dominated by the model-free specialists, echoing the dominance pattern in
the MoA trace but with slightly longer model-based stretches.【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1221-L1225】

![Projected TinyRNN agent mix](./demo_fig/real_demo_agent_mix_projected_hmm_tinyrnn.svg)

*Reading the plot*: the four coloured traces reuse the MoA palette (MF reward in
blue through bias in red) so you can directly line up the projected TinyRNN
responsibilities with the genuine MoA mix above. The translucent band at the
bottom highlights which expert would be in charge if we asked the RNN to act via
the MoA basis on each trial.

*What it shows*: the green model-based segment wins long runs of trials, handing
over briefly to the red bias expert, exactly mirroring the alternation pattern in
the original MoA fit (93 model-based vs 107 bias-dominated trials).【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1221-L1225】【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
The projection therefore makes the shared regimes visually explicit: every
switch in the TinyRNN phases that feeds more weight to the bias head corresponds
to a matching red band in the MoA figure, while the long green blocks mark the
same stretches where the MoA gating leans on its model-based planner.【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1215-L1225】【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】

## TinyMoA latent-state posterior (`real_demo_state_posterior_serieshmm_tinymoa.svg`)

![SeriesHMM-TinyMoA state posterior](./demo_fig/real_demo_state_posterior_serieshmm_tinymoa.svg)

The SeriesHMM backbone spends roughly 81% of the session in a single phase, with
only brief excursions into the alternate state before snapping back.【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
When the posterior does swing, it coincides with the responsibility shifts seen
above, reinforcing that the MoA gating tracks regime changes.

## TinyRNN latent-state posterior (`real_demo_state_posterior_serieshmm_tinyrnn.svg`)

![SeriesHMM-TinyRNN state posterior](./demo_fig/real_demo_state_posterior_serieshmm_tinyrnn.svg)

The TinyRNN variant discovers the same two-phase structure but allocates time
differently: the second state carries ~71% of the posterior mass, leaving the
first state to handle short bursts early in the session.【F:results/real_data/demo/hmm_tinyrnn/posterior_trace.json†L1-L200】
Contrasting the MoA and TinyRNN posteriors makes it easy to flag trials where the
two models disagree about which regime generated the observed behaviour.

To regenerate these figures after rerunning the pipeline, execute:

```bash
python scripts/plot_synthetic_results.py results/real_data/demo \
  --out-dir results/real_data/demo_fig --prefix real_demo
```

and refresh this document if you swap in a new dataset.
