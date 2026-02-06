# codexs_failure_to_fix

Date: 2026-02-06

This document summarizes, **separately for Colab and MPS**, the fixes attempted, the outcomes observed, and where I failed to resolve the issue. It is intended to hand off to another agent.

---

## Colab (VCTK, CUDA, T4)

### Baseline State
- Dataset: VCTK cached locally (`./cache/vctk`), 43,327 training samples.
- Model size: ~12.3M parameters (small AR model).
- Config constraints: `max_seq_len=1024`, `max_audio_len=6.0`, `batch_size=1`, `grad_accum_steps=16`.
- Goal: stop **codebook collapse** (usage often ~1–5%) and **attention collapse**.

### Fixes Attempted
1) **Lower LR + longer warmup + lower label smoothing**
   - `lr: 1e-4 → 5e-5`
   - `warmup_steps: 1000 → 2000`
   - `label_smoothing: 0.1 → 0.05`
   - Result: collapse persisted; utilization dropped to ~1–3% by 3–5k steps.

2) **Entropy regularization on logits (early only)**
   - Added in trainer: `codebook_entropy_weight` with linear decay.
   - Initial config: `weight: 0.02`, `warmup_steps: 10000`.
   - Result: no material improvement; utilization still ~1–3%.

3) **Adjusted warmup length**
   - Reduced `warmup_steps` from 2000 to 500 to get LR out of “floor.”
   - Result: LR rose faster, but collapse still happened by ~2k steps.

4) **Stronger entropy + token noise during teacher forcing**
   - Trainer change: random token noise applied to inputs during early steps.
   - Config:
     - `codebook_entropy_weight: 0.12`
     - `codebook_entropy_warmup_steps: 20000`
     - `token_noise_prob: 0.10`
     - `token_noise_warmup_steps: 20000`
   - Result: not validated yet in logs after the change; user reported continued collapse ~2k steps even with previous changes.

### Other Context / Observations
- LR at 2k steps was still extremely low due to warmup + grad accumulation.
- Warnings persisted:
  - `WARNING: Codebook collapse detected! Only 0.8–2.4% of vocab used`
  - `WARNING: Attention collapse detected!`
- Val loss was decreasing, but collapse made runs unusable for quality.
- Eval spam indicated Colab sometimes running **old code**, requiring `git pull` + runtime restart.

### Open Questions / Likely Root Causes
- **Effective optimizer update frequency** is too low (grad_accum=16) for discrete codebook modeling.
- LR schedule may not match discrete token dynamics; may need **constant LR after short warmup**.
- Collapse might require **token dropout**, **sampling temperature in logits**, or **explicit KL regularization** against uniform.
- Dataset/tokenization mismatch possible (not yet proven).

### Failure Summary (Colab)
I failed to stop codebook collapse; utilization remained ~1–5% despite LR reductions, entropy regularization, and token noise. The system appears stuck in a degenerate minimum very early. I did not deliver a stable run.

---

## MPS (Mac, LJSpeech)

### Baseline State
- Dataset: LJSpeech cached (`./cache/ljspeech`), ~12.8k samples.
- Model size: ~5.9M parameters (small AR model).
- Config constraints: `max_seq_len=1024`, `max_audio_len=4.0`, `batch_size=1`, `grad_accum_steps=32`.

### Fixes Attempted
1) **Lower LR + longer warmup + lower label smoothing**
   - `lr: 1e-4 → 3e-5`
   - `warmup_steps: 100 → 500`
   - `label_smoothing: 0.2 → 0.05`
   - Result: utilization started ~23% but dropped to 12% by ~4k steps.

2) **Entropy regularization (moderate)**
   - `codebook_entropy_weight: 0.03 → 0.06`
   - `codebook_entropy_warmup_steps: 15000 → 20000`
   - Result: still collapsed downward by ~4–5k steps.

3) **Token noise during teacher forcing**
   - Added: `token_noise_prob: 0.10`, `token_noise_warmup_steps: 20000`.
   - Result: not resolved; collapse still at ~6.5% by ~5k steps.

4) **Planned but not executed**
   - Reduce `grad_accum_steps` (32 → 8) and raise LR to restore optimizer update frequency.
   - This was suggested but not implemented before user stopped.

### Failure Summary (MPS)
I failed to prevent collapse; utilization still fell below ~10% by 5k steps even after entropy regularization and token noise. The run was still degenerating.

---

## Code / Config Changes Attempted

Trainer modifications:
- Added **entropy regularization** to AR loss.
- Added **token noise** applied to teacher‑forcing inputs.
- Exposed scheduler type / min LR config.

Configs modified:
- `configs/experiment/vctk_colab.yaml` (multiple LR/warmup/entropy iterations)
- `configs/experiment/ljspeech_mps.yaml` (LR/warmup/entropy/token noise)

---

## What Another Agent Should Try Next (Suggested)

### For Colab
- Reduce `grad_accum_steps` (16 → 4 or 8) to increase update frequency.
- Use **constant LR after short warmup** (e.g., warmup 200 steps; constant LR).
- Add **token dropout** (masking a % of inputs to a learned mask token).
- Add **KL regularization** to a uniform prior or target entropy floor.
- Consider **smaller max_seq_len** (e.g., 768) to reduce collapse pressure.

### For MPS
- Reduce `grad_accum_steps` (32 → 8 or 4), increase LR accordingly.
- Shorten `warmup_steps` to match new accum ratio.
- Test with `max_audio_len` shorter (e.g., 3.0s).

---

## Final Note
I did not deliver a stable, non‑collapsed training run on either platform. This file is a complete summary of what I tried and what failed, so another agent can take over without repeating the same steps.
