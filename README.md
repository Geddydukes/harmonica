# Harmonica: Micro Neural Voice Cloning System

A research-grade neural voice cloning system following the VALL-E codec-LM paradigm, trainable on constrained hardware (24GB Mac Mini + Colab free tier).

## Features

- **VALL-E Architecture**: AR + NAR transformer design for high-quality TTS
- **Zero-Shot Voice Cloning**: Clone voices from 3-10 seconds of reference audio
- **Neural Audio Codec**: EnCodec integration for high-fidelity audio tokenization
- **Portable Training**: Seamless checkpoint transfer between Mac (MPS) and Colab (CUDA)
- **Efficient Design**: ~50M parameter AR model, ~30M parameter NAR model
- **Audio Preprocessing**: Standardized resample/trim/normalize for consistent codec input
- **Scheduled Sampling**: Reduces exposure bias during AR training
- **Length Control**: Duration predictor, stop token, or length prompt modes
- **Monitoring & Failure Detection**: Codebook utilization, attention entropy, gradient stats, and eval checks
- **Full Reproducibility**: RNG + sampler state captured in checkpoints

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/harmonica.git
cd harmonica

# Install dependencies
pip install -e ".[dev]"
```

### Download Data

```bash
# Download LJSpeech (recommended for initial experiments)
make download-ljspeech

# Or download manually from:
# https://keithito.com/LJ-Speech-Dataset/

# Download VCTK (uses aria2c if available for faster download)
make download-vctk
```

### Preprocess Data

```bash
# Encode audio to codec tokens (caches for fast training)
make preprocess-ljspeech
```

### Hugging Face VCTK (cached, auto-cleanup)

You can stream VCTK directly without downloading the full dataset:

```bash
# Train with HF cache (cleared at end)
bash scripts/train_vctk_stream.sh auto /tmp/hf_vctk_cache
```

This uses the Hugging Face `vctk` dataset by default and removes the cache directory after training.
If you want a different HF dataset, update `hf_dataset` in `configs/experiment/vctk_stream.yaml`.

### Train

```bash
# Train AR model on LJSpeech
python scripts/train.py --config configs/experiment/baseline.yaml

# Resume training
python scripts/train.py --config configs/experiment/baseline.yaml --resume
```

### Synthesize

```bash
# Generate speech from text
python scripts/synthesize.py \
    --text "Hello, this is a test of the voice cloning system." \
    --ar-checkpoint experiments/checkpoints/checkpoint_best.pt \
    --output output.wav

# With voice cloning from reference
python scripts/synthesize.py \
    --text "Hello, this is a test." \
    --ar-checkpoint experiments/checkpoints/checkpoint_best.pt \
    --reference reference_audio.wav \
    --output cloned_voice.wav
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         HARMONICA                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT:  "Hello world" + 5s reference audio                     │
│                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │ Text Encoder   │  │ Speaker Encoder│  │ Audio Codec    │    │
│  │ (char → emb)   │  │ (ref → embed)  │  │ (wav ↔ tokens) │    │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘    │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                AR Transformer (~50M)                      │  │
│  │  • Predicts codebook 1 tokens autoregressively           │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                NAR Transformer (~30M)                     │  │
│  │  • Predicts codebooks 2-8 in parallel                    │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                Codec Decoder (Frozen)                     │  │
│  │  • Converts tokens → 24kHz waveform                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  OUTPUT: Generated waveform matching reference speaker          │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
harmonica/
├── configs/                  # Configuration files
│   ├── config.yaml           # Base config
│   ├── model/                # Model configs
│   ├── data/                 # Dataset configs
│   └── experiment/           # Experiment configs
├── harmonica/                # Main package
│   ├── codec/                # Audio codec (EnCodec, DAC)
│   ├── text/                 # Text tokenization
│   ├── data/                 # Data loading
│   ├── model/                # Model architecture
│   ├── training/             # Training infrastructure
│   ├── inference/            # Generation/synthesis
│   └── utils/                # Utilities
├── scripts/                  # CLI scripts
├── tests/                    # Unit tests
├── eval/                     # Evaluation assets
└── notebooks/                # Colab notebooks
```

## Training on Different Hardware

### Mac (MPS)

```bash
python scripts/train.py --device mps
```

### Colab (CUDA)

```bash
python scripts/train.py --device cuda
```

### Checkpoint Portability

Checkpoints are automatically portable between devices:

```python
# Load checkpoint on any device
from harmonica.training import load_checkpoint

# Will automatically map to current device
load_checkpoint("checkpoint.pt", model=model, device=torch.device("mps"))
```

## Datasets

| Dataset | Type | Size | Speakers |
|---------|------|------|----------|
| LJSpeech | Single | ~24h | 1 |
| VCTK | Multi | ~44h | 110 |
| LibriTTS | Multi | ~500h | 2456 |

## Configuration

Key configuration options in `configs/config.yaml`:

```yaml
model:
  ar:
    d_model: 512
    n_heads: 8
    n_layers: 12
    length_control_mode: "duration_predictor"
    duration_hidden_dim: 256

training:
  batch_size: 8
  grad_accum_steps: 4
  lr: 1e-4
  max_steps: 100000
  scheduled_sampling: true
  teacher_forcing_schedule: "linear"
  teacher_forcing_start: 1.0
  teacher_forcing_end: 0.3
  duration_loss_weight: 0.1

inference:
  temperature: 0.8
  top_k: 50
  top_p: 0.95
```

### Length Control Modes

- `duration_predictor`: Predicts target token length from text encodings.
- `stop_token`: Adds a special token and stops on it (requires training to include stop token).
- `length_prompt`: Embeds a coarse length estimate into text conditioning.

### Evaluation Failure Detection

`scripts/evaluate.py` runs signal checks (silence, clipping, looping, noise) and optional ASR-based gibberish detection.
Use `--enable-gibberish` to enable Whisper-based checks (slower). Results are in `results.json` under `failure_rates`.

## Development

```bash
# Run tests
make test

# Format code
make format

# Lint
make lint

# Smoke test (verify imports)
make smoke-test
```

## References

- [VALL-E](https://arxiv.org/abs/2301.02111): Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers
- [EnCodec](https://arxiv.org/abs/2210.13438): High Fidelity Neural Audio Compression
- [DAC](https://arxiv.org/abs/2306.06546): High-Fidelity Audio Compression with Improved RVQGAN

## License

MIT License
