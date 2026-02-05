.PHONY: install test lint format clean download preprocess train synth eval

# Installation
install:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=harmonica --cov-report=html

# Linting and formatting
lint:
	ruff check harmonica/ scripts/ tests/
	black --check harmonica/ scripts/ tests/

format:
	black harmonica/ scripts/ tests/
	ruff check --fix harmonica/ scripts/ tests/

# Data preparation
download-ljspeech:
	python scripts/download_data.py --dataset ljspeech --output-dir ./data

download-vctk:
	python scripts/download_data.py --dataset vctk --output-dir ./data

download-libritts:
	python scripts/download_data.py --dataset libritts --output-dir ./data

preprocess-ljspeech:
	python scripts/preprocess.py --dataset ljspeech --data-dir ./data/LJSpeech-1.1 --output-dir ./cache/ljspeech

preprocess-vctk:
	python scripts/preprocess.py --dataset vctk --data-dir ./data/VCTK-Corpus --output-dir ./cache/vctk

preprocess-vctk-stream:
	@echo "Streaming VCTK does not require preprocessing. Use 'make train-vctk-stream'."

# Training
train-ar:
	python scripts/train.py --config configs/experiment/baseline.yaml --model-type ar

train-nar:
	python scripts/train.py --config configs/experiment/baseline.yaml --model-type nar

train-resume:
	python scripts/train.py --config configs/experiment/baseline.yaml --model-type ar --resume

train-ljspeech:
	python scripts/download_data.py --dataset ljspeech --output-dir ./data
	PYTHONPATH=. .venv/bin/python scripts/preprocess.py --dataset ljspeech --data-dir ./data/LJSpeech-1.1 --output-dir ./cache/ljspeech
	PYTHONPATH=. .venv/bin/python scripts/train.py --config configs/experiment/baseline.yaml --model-type ar

train-vctk-base:
	bash scripts/train_vctk.sh ./data/VCTK-Corpus ./cache/vctk auto

train-vctk-stream:
	KEEP_CACHE=1 PYTHON_BIN=.venv/bin/python bash scripts/train_vctk_stream.sh auto /tmp/hf_vctk_cache

# Synthesis
synth:
	@echo "Usage: make synth TEXT='Hello world' OUTPUT=output.wav CHECKPOINT=path/to/checkpoint.pt"
	python scripts/synthesize.py --text "$(TEXT)" --output "$(OUTPUT)" --ar-checkpoint "$(CHECKPOINT)"

# Evaluation
eval:
	python scripts/evaluate.py --ar-checkpoint experiments/checkpoints/checkpoint_best.pt --output-dir eval/output

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf harmonica/__pycache__ harmonica/**/__pycache__
	rm -rf tests/__pycache__
	rm -rf *.egg-info build dist
	rm -rf experiments/logs/* experiments/checkpoints/*

clean-cache:
	rm -rf cache/*

# Development helpers
check-device:
	python -c "from harmonica.utils.device import device_info; print(device_info())"

count-params:
	python -c "\
from harmonica.model import ARTransformer; \
m = ARTransformer(); \
print(f'AR params: {sum(p.numel() for p in m.parameters()):,}')"
	python -c "\
from harmonica.model import NARTransformer; \
m = NARTransformer(); \
print(f'NAR params: {sum(p.numel() for p in m.parameters()):,}')"

# Quick smoke test
smoke-test:
	python -c "\
from harmonica.codec import EnCodecBackend; \
from harmonica.text import CharTokenizer; \
from harmonica.model import ARTransformer; \
import torch; \
codec = EnCodecBackend(); \
tok = CharTokenizer(); \
model = ARTransformer(text_vocab_size=tok.vocab_size); \
print('Codec:', codec.sample_rate, 'Hz,', codec.n_codebooks, 'codebooks'); \
print('Tokenizer:', tok.vocab_size, 'vocab'); \
print('AR Model:', sum(p.numel() for p in model.parameters()), 'params'); \
print('All imports successful!')"
