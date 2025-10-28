# Synthetic HMM-MoA vs TinyRNN Process

This note documents how the repository synthesises behavioural sequences, why the design mirrors the HMM-MoA paper, and how the two competing emission models are trained. It focuses on the ingredients of the workflow (data, model, optimisation) rather than the final score tables.

![Generator overview](fig/synthetic_generation.svg?raw=1)

## 1. Synthetic data generation

### 1.1 Objectives and rationale
- **Match the HMM-MoA stimulus** – sequences emulate the two-step task with alternating “common” and “rare” transitions so that the HMM can infer latent phases representing model-free vs. model-based control modes.
- **Encourage long mode persistence** – sticky transitions create sustained blocks of each latent phase, making recovery by both HMM variants meaningful.
- **Provide ground-truth supervision** – we record the latent phase labels alongside observable actions, rewards, and transition categories to validate phase recovery.

### 1.2 Generator pipeline
1. **Phase sampling:** We initialise each sequence with a random phase and then draw contiguous blocks whose length is centred on the requested dwell time (default 120) with ±10 steps of jitter. This mirrors the quasi-deterministic regime used in the HMM-MoA work.【F:series_hmm_rnn/data.py†L4-L18】
2. **Reward templates:** Each phase selects between two reward maps `R0` and `R1` that favour opposite first-stage actions, yielding a pair of Q-value templates conditioned on the latent phase.【F:series_hmm_rnn/data.py†L12-L16】
3. **Policy mixture:** A softmax with inverse temperature `β` (default 3.5) samples actions from the mixture Q-values, blending common and rare transition expectations as in the task definition.【F:series_hmm_rnn/data.py†L16-L19】
4. **Outcome sampling:** We simulate rare/common transitions and rewards based on the chosen action and latent state, returning `(actions, rewards, transitions, latent_states)` tensors for each batch element.【F:series_hmm_rnn/data.py†L19-L25】

The generator is invoked twice—once for training, once for testing—with independent seeds to avoid leakage between splits.【F:series_hmm_rnn/run_synthetic_pipeline.py†L88-L102】

## 2. Training workflow

![Model comparison](fig/model_comparison.svg?raw=1)

### 2.1 Shared encoder and optimiser
- A compact GRU encoder with hidden size 6 processes the action–reward–transition history for both emission heads.【F:series_hmm_rnn/models.py†L164-L199】【F:series_hmm_rnn/models.py†L247-L279】
- We train both models with Adam (`lr = 1e-3`) for a user-selected number of epochs (150 in the main experiment) on mini-batches comprising the entire synthetic dataset.【F:series_hmm_rnn/run_synthetic_pipeline.py†L120-L162】
- The HMM parameters are initialised with a sticky transition matrix (stay probability 0.97) and small random symmetry breakers to encourage mode persistence, following the MoA paper’s strategy.【F:series_hmm_rnn/run_synthetic_pipeline.py†L21-L35】【F:series_hmm_rnn/run_synthetic_pipeline.py†L122-L129】

### 2.2 HMM-MoA emission head
- **Agent ensemble:** Four agents (Model-free value, Model-free choice, Model-based, and Bias) produce candidate Q-values, reproducing the mixture-of-agents formulation.【F:series_hmm_rnn/run_synthetic_pipeline.py†L13-L19】
- **Gating network:** A learned linear head transforms the GRU state into agent weights (`Wg·h + b`) whose softmax drives the mixture; the resulting aggregate Q-values feed the HMM observation model.【F:series_hmm_rnn/models.py†L215-L245】
- **Training target:** The negative log-likelihood combines the HMM forward–backward log probabilities with the emitted policy to match observed actions, while accuracy tracks correct first-stage choices.【F:series_hmm_rnn/train.py†L128-L188】

### 2.3 HMM-TinyRNN emission head
- **Direct readouts:** Instead of mixing handcrafted agents, this head attaches per-phase linear projections on top of the shared GRU state, letting the model learn smooth Q-functions directly.【F:series_hmm_rnn/models.py†L247-L279】
- **No auxiliary agents:** Training omits the agent feature stack; only the GRU state and per-phase heads determine the emission probabilities.【F:series_hmm_rnn/run_synthetic_pipeline.py†L130-L162】
- **Identical HMM inference:** The same forward–backward machinery and loss calculation ensure that differences in performance come solely from the emission model rather than the latent-state dynamics.【F:series_hmm_rnn/train.py†L128-L188】

## 3. Assumptions and design choices
- **Two latent phases (`K=2`):** Mirrors the binary regime of the original task (habitual vs. goal-directed) and keeps permutation alignment manageable.【F:series_hmm_rnn/run_synthetic_pipeline.py†L110-L117】
- **Single long sequence per batch:** We generate long trajectories (`T=400`) and train with full-length batches to emphasise temporal credit assignment rather than stochastic mini-batching.【F:series_hmm_rnn/run_synthetic_pipeline.py†L100-L117】
- **Sticky transitions:** A 0.97 stay probability biases the HMM towards long dwell times, matching the synthetic generator’s behaviour and stabilising alignment across runs.【F:series_hmm_rnn/run_synthetic_pipeline.py†L21-L35】【F:series_hmm_rnn/run_synthetic_pipeline.py†L122-L138】
- **Shared temperature (`τ=1.25`):** Both heads share the inverse-temperature used to convert Q-values into choice probabilities, allowing a fair comparison between mixture-based and smooth RNN emissions.【F:series_hmm_rnn/run_synthetic_pipeline.py†L110-L117】
- **Evaluation metrics:** Phase accuracy uses a permutation-invariant alignment with the ground truth labels captured during simulation, ensuring that latent state identities are comparable even when label-swapped.【F:series_hmm_rnn/run_synthetic_pipeline.py†L51-L77】

## 4. Reproducing the process
1. `python -m series_hmm_rnn.run_synthetic_pipeline --epochs 150 --B 64 --T 400 --out-dir results/synthetic_run1`
2. Generate the conceptual figures above (already tracked under `fig/`).
3. Review `results/results.md` and `results/visualizations.md` for quantitative summaries once the run completes.

This document captures the reasoning and components underpinning the synthetic benchmarking workflow so it can be understood independently of the final metrics.
