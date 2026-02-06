# Harmonica Colab Training – Root Cause Analysis (VCTK)

Date: 2026-02-06

This doc summarizes the observed failures and collapse behaviors in Colab training, the underlying root causes, and the consolidated fixes/steps to stabilize training.

## Executive Summary
Training is **stable but degenerate**: loss decreases while **codebook utilization collapses** (<5–10%) and **attention diversity collapses**. This is a classic symptom of high‑entropy token modeling collapsing into a small subset of tokens under overly aggressive optimization and insufficient regularization.

To fix it, the training configuration and loop need to explicitly discourage early collapse, slow down the optimizer, and ensure length constraints are consistent. A consolidated patch provides:

- **Entropy regularization on codebook logits (early only)**
- **Lower LR + longer warmup**
- **Lower label smoothing**
- **Consistent max audio length vs max sequence length**
- **Scheduler type/min LR configurable**

## Observed Symptoms
- **Codebook collapse warnings** at <5–10% utilization for long stretches.
- **Attention collapse warnings** consistently in early steps.
- **Previously seen shape errors** (1025 vs 1024) tied to max_seq_len / stop_token length handling.
- **Eval spam** shown in Colab logs even though code was updated locally (indicates old code in Colab).

## Root Causes
### 1) Aggressive Optimization + Insufficient Early Regularization
- Colab runs used **LR=1e‑4** with small batch (effective batch = 16 via grad accumulation but still noisy), **label_smoothing=0.1**, and **short warmup**.
- This leads to **high‑confidence early collapse** into a tiny subset of codebook tokens.

### 2) No Explicit Anti‑Collapse Mechanism
- The model had no objective term encouraging broad usage of the codebook.
- Without an entropy penalty, the easiest path to lower loss is to over‑predict frequent tokens.

### 3) Sequence Length Mismatch & Stop Token Edge
- Earlier runs showed `1025 vs 1024` and `1027 vs 1024` errors.
- Root cause was **stop token appended** while positional encodings were only built to `max_seq_len`.
- Fix is to **cap audio sequence length to max_seq_len‑1** and clamp positional encoding.

### 4) Scheduler Type Ignored
- `scheduler_type` in configs was defined but **not passed** to the scheduler factory.
- This made it hard to control LR dynamics across platforms.

### 5) Colab Not Running Updated Code
- Colab logs still show evaluation tqdm output (which was removed locally).
- This indicates **the repo in Colab wasn’t pulled**, so fixes weren’t applied.

## Consolidated Fixes (applied in repo)
### Training Loop
- **Entropy regularization** on codebook logits early in training (decays to 0 by step 10k).
- **Scheduler type/min LR now configurable**.

Files:
- `harmonica/training/trainer.py`

### Colab Config (VCTK)
- LR lowered: `1e‑4 → 5e‑5`
- Warmup increased: `1000 → 2000`
- Label smoothing lowered: `0.1 → 0.05`
- Added entropy regularization:
  - `codebook_entropy_weight: 0.02`
  - `codebook_entropy_warmup_steps: 10000`

File:
- `configs/experiment/vctk_colab.yaml`

## Consolidated Colab Steps (exact)

### 1) Restart runtime (important)
Colab often keeps old code in memory. Do a **Runtime → Restart runtime**.

### 2) Pull latest repo and install
```bash
!git pull
!pip install -e .
```

### 3) Verify you’re on the updated code
```bash
!grep -n "codebook_entropy" -n harmonica/training/trainer.py
!grep -n "tqdm(self.val_loader" -n harmonica/training/trainer.py
```

### 4) Run training
```bash
!python scripts/train.py --config configs/experiment/vctk_colab.yaml --model-type ar --device cuda
```

### 5) Watch these signals
- **Codebook utilization** should climb toward **20–40%** after a few thousand steps.
- **Attention entropy** should stabilize; warnings should fade.
- **Val loss** should track train loss without divergence.

If utilization remains <10% after ~5–10k steps, we need stronger regularization (next step: increase entropy weight or add token‑drop/temperature noise).

## If Collapse Persists (Next Options)
In priority order:
1. Increase entropy weight to `0.04` and extend warmup to `20000`.
2. Reduce LR further to `3e‑5`.
3. Enable scheduled sampling **after** warmup (start at step 10k with ratio 0.95→0.7).
4. Add mild token‑noise to teacher forcing inputs (only if needed).

---

## Quick Checklist
- [ ] Runtime restarted
- [ ] `git pull` done
- [ ] `pip install -e .` done
- [ ] Using `configs/experiment/vctk_colab.yaml`
- [ ] Training logs show codebook utilization increasing

