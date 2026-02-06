# Restructure Plan For Higher Model Quality

Date: 2026-02-06  
Scope: Full repository plan for materially better synthesis quality, stability, and iteration speed.

## 1) Goals And Success Criteria

### Primary goals
- Increase naturalness and intelligibility of generated speech.
- Eliminate persistent codebook/attention collapse patterns.
- Improve robustness across MPS (local) and CUDA (Colab/cloud).
- Make training/evaluation loops faster to diagnose and less wasteful.

### Success criteria (measurable)
- NAR per-codebook predicted utilization stabilizes above configured floor (for example, median per-codebook relative utilization > 25% after warmup).
- Evaluation failure rates stay near zero for silence/clipping/looping/noise on held-out prompts.
- MOS proxy metrics trend better run-over-run (ASR WER/CER on generated speech, speaker similarity consistency).
- Best checkpoint selection is reproducible and automatically reflected in metadata.

Why: The repo currently optimizes losses but still yields robotic outputs and collapse warnings. We need target metrics that correlate with real quality, not only CE/perplexity.

---

## 2) High-Leverage Model Changes

## 2.1 Add explicit speaker conditioning in NAR
- Add a `SpeakerEncoder` path for NAR, parallel to text conditioning.
- During training, feed a short reference prompt (or learned speaker embedding per speaker for single-speaker experiments).
- In NAR blocks, condition via cross-attention on concatenated context (`text_emb + speaker_emb`) or dual cross-attention streams.

Why:
- Current NAR path is under-conditioned and drifts toward average/robotic outputs.
- VALL-E style quality depends on robust speaker/prosody conditioning in both AR and NAR stages.

Concrete repo changes:
- `harmonica/model/nar.py`: add speaker-conditioning inputs and modules.
- `harmonica/data/collate.py`: ensure consistent prompt token extraction for NAR training batches.
- `scripts/train.py` and configs: speaker conditioning flags and prompt lengths.

## 2.2 Replace plain NAR CE with confidence-aware multi-term objective
- Keep token CE, but add:
- Codebook entropy regularization (already present): keep.
- Marginal usage entropy regularization (already added): keep.
- Confidence penalty / label smoothing schedule per codebook (heavier early, decay later).
- Optional focal-style token weighting for overrepresented tokens.

Why:
- Pure CE on discrete codec tokens tends to collapse into frequent-token basins.
- A staged objective widens exploration early then sharpens late.

Concrete repo changes:
- `harmonica/model/nar.py`: objective term composition, per-codebook weights.
- `configs/*`: `loss_weights` section with schedule knobs.

## 2.3 Add NAR iterative refinement training mode
- Train with 1-step and 2-step refinement passes (teacher-forced then partially self-conditioned).
- At inference, support configurable `nar_refine_steps`.

Why:
- One-pass NAR decoding is brittle and often sounds quantized/robotic.
- Refinement improves harmonic detail and smoothness with limited compute overhead.

Concrete repo changes:
- `harmonica/model/nar.py`: refinement forward variants.
- `harmonica/inference/nar_decode.py`: iterative decode path with quality/perf switch.

## 2.4 Improve AR->NAR interface contract
- Freeze and log AR interface assumptions in checkpoint metadata:
- `max_seq_len`, codebook mapping, token rate, BOS/EOS handling, text encoder dimensions.
- Add hard compatibility check before loading NAR with AR at eval/synthesis.

Why:
- Silent AR/NAR incompatibilities create subtle quality failures and wasted runs.

Concrete repo changes:
- `harmonica/training/checkpoint.py`: explicit interface schema.
- `scripts/evaluate.py`, `scripts/synthesize.py`: strict compatibility validation.

---

## 3) Data And Token Pipeline Restructure

## 3.1 Rebuild/validate cache deterministically per experiment
- Cache fingerprint should include:
- codec settings, sample rate, max/min duration, trim/normalize settings, model seq limits.
- Refuse stale cache by default unless `--allow-cache-mismatch` is explicitly set.

Why:
- Cache mismatch is a hidden source of truncation artifacts and shape/pathology drift.

Concrete repo changes:
- `scripts/preprocess.py` and cache metadata schema.
- `scripts/train.py` cache mismatch logic: hard-fail option.

## 3.2 Improve data quality filtering
- Add optional dataset filtering by:
- long silences, clipped audio ratio, low SNR heuristics, transcript length mismatch.
- Produce a preprocessing quality report (`json/csv`) and blacklist file.

Why:
- NAR learns token distribution from codec targets; noisy/outlier audio amplifies collapse.

Concrete repo changes:
- `harmonica/utils/audio.py` + preprocessing scripts.
- `data/` metadata generation pipeline.

## 3.3 Balanced sampling strategy
- Implement sampler that balances:
- utterance lengths, speakers (for multispeaker), and phoneme/character coverage buckets.

Why:
- Imbalanced mini-batches bias token marginals and accelerate collapse.

Concrete repo changes:
- `harmonica/data/sampler.py`: add stratified/bucket sampler.
- config support in `configs/*`.

---

## 4) Training Loop And Optimization

## 4.1 Split AR and NAR training configs clearly
- Move shared config into base; create separate explicit AR and NAR training blocks.
- Avoid reused knobs with ambiguous semantics.

Why:
- AR and NAR failure modes differ; shared knobs hide wrong assumptions.

Concrete repo changes:
- `configs/config.yaml`: `training.ar` and `training.nar` sections.
- `scripts/train.py`: select block by `--model-type`.

## 4.2 Add optimizer parameter groups
- Use separate LR/weight decay groups for:
- embeddings, attention blocks, output projections, text encoder, optional speaker encoder.

Why:
- Discrete token heads often need different dynamics from context encoders.

Concrete repo changes:
- `harmonica/training/optimizer.py`: param-group factory.

## 4.3 Add EMA weights for eval/checkpointing
- Maintain EMA shadow model and evaluate/save both raw and EMA.
- Prefer EMA for synthesis/eval by default.

Why:
- EMA usually yields smoother generations and less noisy checkpoint ranking.

Concrete repo changes:
- `harmonica/training/trainer.py`: EMA updates and save/load.
- `harmonica/training/checkpoint.py`: EMA state support.

## 4.4 Add gradient health safeguards
- Track per-module gradient norms and update/weight ratios.
- Auto-reduce LR on repeated instability events.

Why:
- Collapse often starts with unstable or vanishing module-specific updates.

Concrete repo changes:
- `harmonica/training/metrics.py` and trainer callbacks.

## 4.5 Early stopping and plateau policies
- Add optional early-stop on:
- no `val_loss` improvement and no utilization improvement window.
- Add plateau scheduler fallback.

Why:
- Avoid burning compute on runs that converge to degenerate minima.

---

## 5) Diagnostics And Evaluation Overhaul

## 5.1 Promote per-codebook metrics to first-class outputs
- Log and print:
- per-codebook predicted/target utilization,
- per-codebook usage entropy,
- per-codebook CE loss trends.

Why:
- Global utilization hides which codebooks are failing.

Concrete repo changes:
- Extend `harmonica/training/metrics.py` (started; expand dashboards/alerts).

## 5.2 Add objective-aligned eval suite
- Expand `scripts/evaluate.py` to include:
- ASR WER/CER on generated audio,
- speaker embedding similarity (for cloning),
- prosody proxies (duration/pitch variance consistency),
- batch-level summary report for regression tracking.

Why:
- Current eval detects obvious failures but not “robotic but non-failing” audio.

## 5.3 Structured experiment reports
- Save per-run markdown report with:
- config diff, key curves, best checkpoint info, eval summary, failure reasons.

Why:
- Prevent repeated “roundabout” cycles and preserve decisions.

Concrete repo changes:
- `scripts/report_experiment.py` + output under `experiments/reports/`.

---

## 6) Inference Path Improvements

## 6.1 Better decode controls
- Add per-codebook temperature/top-k/top-p defaults.
- Add optional repetition penalty and anti-loop heuristics for AR.
- Add NAR refine-step parameter exposed in CLI.

Why:
- Single global temperature is too coarse; codebooks have different entropy needs.

Concrete repo changes:
- `harmonica/inference/ar_decode.py`
- `harmonica/inference/nar_decode.py`
- `scripts/synthesize.py` CLI args.

## 6.2 Real-time compatibility checks at load time
- Block synthesis if AR/NAR interface signatures mismatch.
- Print actionable remediation message.

Why:
- Prevent silent quality degradation from incompatible checkpoints.

---

## 7) Repository Structure And Engineering Hygiene

## 7.1 Separate research code from productionized paths
- Create:
- `harmonica/research/` for experimental objectives,
- `harmonica/core/` stable training/inference APIs.

Why:
- Fast experimentation without destabilizing baseline training path.

## 7.2 Add strict typed config schema
- Pydantic/dataclass schema with validation and explicit defaults.
- Fail fast for unknown/misplaced keys.

Why:
- YAML drift and typo-prone knobs currently cost time and cause hidden behavior.

Concrete repo changes:
- New `harmonica/config/schema.py`.
- Validate in `scripts/train.py`, `scripts/evaluate.py`, `scripts/synthesize.py`.

## 7.3 Stronger tests for training dynamics
- Add tests for:
- collapse warning thresholds,
- best-checkpoint metadata updates,
- AR/NAR compatibility checks,
- schedule conversion edge cases across `grad_accum_steps`.

Why:
- Current unit tests are good on shape/forward behavior but can expand on dynamics and run control.

---

## 8) Prioritized Implementation Plan

## Phase 0 (already partly done)
- NAR text encoder unification.
- NAR anti-collapse regularizers.
- per-codebook collapse logging.
- NAR output dir isolation.

## Phase 1 (highest immediate ROI)
- Add speaker conditioning to NAR.
- Add AR/NAR compatibility contracts and hard checks.
- Add EMA and best-checkpoint tracking improvements.
- Expand eval with ASR WER/CER and speaker-sim score.

Expected impact: Largest quality gain for current architecture with moderate effort.

## Phase 2 (next quality step)
- NAR iterative refinement training/inference.
- Balanced sampler and stricter data filtering.
- Param-group optimizer strategy and plateau fallback.

Expected impact: Better realism and robustness, fewer collapse runs.

## Phase 3 (structural modernization)
- Typed config schema.
- research/core code split.
- run report generation and experiment registry.

Expected impact: Much faster iteration and lower regression risk.

---

## 9) Proposed Config Baseline (Next serious run)

For NAR on LJSpeech MPS:
- Keep `lr: 6e-5` (better diversity than 4e-5 in recent runs).
- Keep `nar_scheduled_sampling: true`.
- Keep entropy + usage-entropy + conditioning-noise warmups through full run.
- Keep per-codebook logging and use early-stop on both loss and utilization plateaus.

For Colab:
- Keep slightly softer `nar_teacher_forcing_end` and usage-entropy weights initially.
- Same diagnostics and stop criteria.

Why:
- Current evidence shows lower LR improved CE but reduced diversity.
- Collapse mitigation needs sustained exploration pressure during warmup window.

---

## 10) Non-Goals (for now)

- Full architecture replacement (diffusion/flow vocoder stack) in this repo.
- Large-scale pretraining on external multi-thousand-hour corpora.
- TPU-specific training path before stabilizing core AR/NAR behavior.

Why:
- These are expensive detours before current failure modes are controlled.

---

## 11) Final Recommendation

Treat the next run as a structured validation run, not just another training run:
- fixed config,
- fixed checkpoints naming,
- fixed eval slices,
- explicit stop criteria,
- written report at the end.

That process change will likely save more time than any single hyperparameter tweak.
