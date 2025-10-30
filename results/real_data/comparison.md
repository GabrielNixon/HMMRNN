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
