(cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/results/real_data/comparison.md b/results/real_data/comparison.md
index 514d75e699f55440603af27be075f1d51453e455..7d4b49326a539629b39242a0e598a16827dce9bd 100644
--- a/results/real_data/comparison.md
+++ b/results/real_data/comparison.md
@@ -1,50 +1,120 @@
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
 
-*What it shows*: in this demo session the model-based and bias experts trade off
-control. The bias head dominates 107 trials while the model-based head wins 93,
-leaving only small windows where the model-free components take charge. Averaged
-across the run the weights settle near 0.16 (MF reward), 0.22 (MF choice), 0.30
-(model-based), and 0.31 (bias), underscoring that the hybrid leans heavily on the
-planning-style and static-bias policies.【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
+*What it shows*: in this demo session the model-based expert now controls the
+majority of trials, claiming 154 of 200, while the MF choice head covers the
+remaining 46; the bias and MF reward experts never become the top-responsibility
+component. Averaged across the run the weights settle near 0.11 (MF reward),
+0.17 (MF choice), 0.43 (model-based), and 0.29 (bias), highlighting a planner-
+versus-choice tug-of-war.【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
+
+## TinyRNN phase responsibility (`real_demo_agent_mix_hmm_tinyrnn.svg`)
+
+![HMM-TinyRNN phase mix](./demo_fig/real_demo_agent_mix_hmm_tinyrnn.svg)
+
+*Reading the plot*: the blue and orange lines track the SeriesHMM posterior over
+the TinyRNN's two recurrent phases, while the coloured bars at the bottom flag the
+ground-truth task blocks for quick alignment with experimental context.
+
+*What it shows*: the first state now averages 0.75 responsibility and wins 149 of
+the 200 trials, leaving the second state with the remaining 0.25 mass across 51
+trials.【F:results/real_data/demo/hmm_tinyrnn/posterior_trace.json†L1-L200】
+Responsibility switches still cluster around the same probe and reversal windows
+that trigger handovers in the TinyMoA gating, signalling that the neural
+controller shadows those behavioural regimes. Correlating each TinyRNN phase with
+the TinyMoA agent mix shows the mapping explicitly: the dominant state co-varies
+with the model-free reward and choice heads (r ≈ 0.51 and 0.50) while
+anti-correlating with the planner and bias experts, whereas the rarer state is
+positively aligned with the model-based and bias responsibilities (r ≈ 0.52 and
+0.48).【F:results/real_data/demo/hmm_tinyrnn/posterior_trace.json†L1-L200】【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
+Together these statistics confirm that the TinyRNN moves through modes that
+shadow the Mixture-of-Agents decomposition rather than inventing entirely new
+dynamics.
+
+## TinyRNN projected agent responsibility (`real_demo_agent_mix_projected_hmm_tinyrnn.svg`)
+
+To make the comparison one-to-one with the TinyMoA breakdown, we project the
+TinyRNN phase posterior onto the MoA agent space by aligning the neural phases to
+the MoA latent states (permutation = [0, 1]) and taking the expectation of the
+MoA per-phase agent weights under the TinyRNN responsibilities. The resulting
+agent mix averages 0.16 MF reward, 0.43 MF choice, 0.27 model-based, and 0.15
+bias responsibility across the session, emphasising the split between the
+model-free choice specialist and the planner.【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1-L1214】
+Trial-by-trial maxima show the projected TinyRNN controller selects the MF choice
+expert on 149 trials and the model-based planner on the remaining 51, with no
+windows dominated by the other heads.【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1-L1214】
+
+![Projected TinyRNN agent mix](demo_fig/real_demo_agent_mix_projected_hmm_tinyrnn.svg)
+
+*Reading the plot*: the four coloured traces reuse the MoA palette (MF reward in
+blue through bias in red) so you can directly line up the projected TinyRNN
+responsibilities with the genuine MoA mix above. The translucent band at the
+bottom highlights which expert would be in charge if we asked the RNN to act via
+the MoA basis on each trial.
+
+*What it shows*: the orange MF-choice band blankets most of the session, while
+the green model-based curve crests over the mid-session reversal block, matching
+the gating pattern seen in the MoA figure. The projection therefore makes the
+shared regimes visually explicit: every switch in the TinyRNN phases that feeds
+more weight to the choice head corresponds to an orange band in the MoA mix,
+while the green ridges mark the same stretches where the MoA gating leans on its
+planner.【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1-L1214】【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
 
 ## TinyMoA latent-state posterior (`real_demo_state_posterior_serieshmm_tinymoa.svg`)
 
 ![SeriesHMM-TinyMoA state posterior](./demo_fig/real_demo_state_posterior_serieshmm_tinymoa.svg)
 
-The SeriesHMM backbone spends roughly 81% of the session in a single phase, with
-only brief excursions into the alternate state before snapping back.【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
-When the posterior does swing, it coincides with the responsibility shifts seen
-above, reinforcing that the MoA gating tracks regime changes.
+The SeriesHMM backbone now spends ~74% of the session in the first phase, with a
+clear second-state block around trials 70–95 and short bursts at the very
+beginning.【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】 The
+stacked area and dominance band highlight those excursions, matching the agent
+switches in the MoA responsibilities above.
 
 ## TinyRNN latent-state posterior (`real_demo_state_posterior_serieshmm_tinyrnn.svg`)
 
 ![SeriesHMM-TinyRNN state posterior](./demo_fig/real_demo_state_posterior_serieshmm_tinyrnn.svg)
 
-The TinyRNN variant discovers the same two-phase structure but allocates time
-differently: the second state carries ~71% of the posterior mass, leaving the
-first state to handle short bursts early in the session.【F:results/real_data/demo/hmm_tinyrnn/posterior_trace.json†L1-L200】
-Contrasting the MoA and TinyRNN posteriors makes it easy to flag trials where the
-two models disagree about which regime generated the observed behaviour.
+The TinyRNN variant retains the same two-phase structure but flips the dwell
+times: the second state now carries ~75% of the posterior mass and dominates the
+late trials, while the first state captures the early block highlighted in the
+dominance band.【F:results/real_data/demo/hmm_tinyrnn/posterior_trace.json†L1-L200】
+Contrasting the MoA and TinyRNN posteriors makes it easy to flag trials where
+the two models disagree about which regime generated the observed behaviour.
 
 To regenerate these figures after rerunning the pipeline, execute:
 
 ```bash
-python scripts/plot_synthetic_results.py results/real_data/demo \
-  --out-dir results/real_data/demo_fig --prefix real_demo
+python scripts/plot_state_posterior.py \
+  results/real_data/demo/hmm_moa/posterior_trace.json \
+  results/real_data/demo_fig/real_demo_state_posterior_serieshmm_tinymoa.svg \
+  --title "SeriesHMM-TinyMoA state posterior"
+
+python scripts/plot_state_posterior.py \
+  results/real_data/demo/hmm_tinyrnn/posterior_trace.json \
+  results/real_data/demo_fig/real_demo_state_posterior_serieshmm_tinyrnn.svg \
+  --title "SeriesHMM-TinyRNN state posterior"
+```
+
+Regenerate the projected TinyRNN agent comparison with:
+
+```bash
+python scripts/plot_projected_agent_mix.py \
+  results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json \
+  results/real_data/demo_fig/real_demo_agent_mix_projected_hmm_tinyrnn.svg \
+  --title "Projected TinyRNN agent responsibilities"
 ```
 
 and refresh this document if you swap in a new dataset.
 
EOF
)
