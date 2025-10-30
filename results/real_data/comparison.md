# SeriesHMM TinyMoA vs TinyRNN — Responsibility Comparison

This note condenses figures under `results/real_data/demo_fig` to show how the hybrids behave internally on the demo real-data run. Each section embeds the generated SVG and highlights slide-ready takeaways.

---

## TL;DR

| View | What | Averages | Dominance counts | Cross-model links |
|---|---|---|---|---|
| **TinyMoA (agent mix)** | MF-R / MF-C / MB / Bias | **0.11 / 0.17 / 0.43 / 0.29** | **MB 154**, **MF-C 46** (Bias/MF-R never top) | Planner–vs–Choice tussle; Bias contributes but rarely leads. |
| **TinyRNN → projected agents** | MF-R / MF-C / MB / Bias | **0.16 / 0.43 / 0.27 / 0.15** | **MF-C 149**, **MB 51** | Projection preserves MoA regimes with smoother transitions. |
| **TinyMoA (state posterior)** | Phase 1 / Phase 2 | **~0.74 / ~0.26** | One clear mid-session Phase-2 block | Agent switches coincide with state excursions. |
| **TinyRNN (state posterior)** | Phase 1 / Phase 2 | **~0.25 / ~0.75** | Phase-2 dominates late trials | Highlights trials where models disagree on regime. |

> Exact numbers and references are cited in the sections below.

---

## TinyMoA agent responsibility (`real_demo_agent_mix_serieshmm_tinymoa.svg`)

![SeriesHMM-TinyMoA agent mix](./demo_fig/real_demo_agent_mix_serieshmm_tinymoa.svg)

**How to read**  
Four trajectories = MoA experts (MF-reward, MF-choice, Model-based, Bias). Values sum to 1 per trial. The band along the bottom marks the most probable expert.

**Key takeaways**
- Model-based controls **154/200** trials; **MF-choice** covers **46/200**. **Bias** and **MF-reward** never become top-responsibility.
- Session averages: **MF-R 0.11**, **MF-C 0.17**, **MB 0.43**, **Bias 0.29** — a planner-vs-choice tug-of-war with a supportive bias component. 【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】

---

## TinyRNN projected agent responsibility (`real_demo_agent_mix_projected_hmm_tinyrnn.svg`)

To compare one-to-one with TinyMoA, we project the TinyRNN phase posterior into the MoA agent space (align phases by permutation `[0, 1]`; take the expectation of MoA per-phase agent weights under TinyRNN responsibilities).

![Projected TinyRNN agent mix](demo_fig/real_demo_agent_mix_projected_hmm_tinyrnn.svg)

**Key takeaways**
- Session averages (projected): **MF-R 0.16**, **MF-C 0.43**, **MB 0.27**, **Bias 0.15** — emphasizing **Choice** vs **Planner**. 【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1-L1214】
- Trial-wise winner (projected): **MF-choice 149** trials, **Model-based 51**; no prolonged MF-reward/Bias dominance. 【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1-L1214】
- Visual alignment: long **orange** (MF-choice) spans and **green** (Model-based) crests mirror MoA bands and RNN phase switches. 【F:results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json†L1-L1214】【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】

---

## TinyMoA latent-state posterior (`real_demo_state_posterior_serieshmm_tinymoa.svg`)

![SeriesHMM-TinyMoA state posterior](./demo_fig/real_demo_state_posterior_serieshmm_tinymoa.svg)

**Key takeaways**
- SeriesHMM backbone spends **~74%** of the session in Phase-1, with a clear Phase-2 block around trials **70–95** and short early bursts. 【F:results/real_data/demo/hmm_moa/posterior_trace.json†L1-L200】
- Excursions match MoA agent switches, reinforcing regime tracking.

---

## TinyRNN latent-state posterior (`real_demo_state_posterior_serieshmm_tinyrnn.svg`)

![SeriesHMM-TinyRNN state posterior](./demo_fig/real_demo_state_posterior_serieshmm_tinyrnn.svg)

**Key takeaways**
- Same two-phase structure but flipped dwell: Phase-2 carries **~75%** of posterior mass and dominates later trials; Phase-1 captures the early block. 【F:results/real_data/demo/hmm_tinyrnn/posterior_trace.json†L1-L200】
- Side-by-side with MoA, this flags trials where the models **disagree** on the generating regime.

---

## Reproduce the figures

```bash
python scripts/plot_state_posterior.py \
  results/real_data/demo/hmm_moa/posterior_trace.json \
  results/real_data/demo_fig/real_demo_state_posterior_serieshmm_tinymoa.svg \
  --title "SeriesHMM-TinyMoA state posterior"

python scripts/plot_state_posterior.py \
  results/real_data/demo/hmm_tinyrnn/posterior_trace.json \
  results/real_data/demo_fig/real_demo_state_posterior_serieshmm_tinyrnn.svg \
  --title "SeriesHMM-TinyRNN state posterior"

python scripts/plot_projected_agent_mix.py \
  results/real_data/demo/hmm_tinyrnn/projected_agent_mix.json \
  results/real_data/demo_fig/real_demo_agent_mix_projected_hmm_tinyrnn.svg \
  --title "Projected TinyRNN agent responsibilities"
