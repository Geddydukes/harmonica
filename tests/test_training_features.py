"""Tests for training features and utilities."""

import torch
import pytest

from harmonica.utils.audio import AudioPreprocessor
from harmonica.training.checkpoint import save_checkpoint, load_checkpoint
from harmonica.training.trainer import Trainer


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return idx


def test_audio_preprocessor_trim_and_normalize():
    pre = AudioPreprocessor(
        target_sample_rate=24000,
        normalization_method="rms",
        target_db=-20.0,
        trim_silence=True,
        silence_threshold_db=-40.0,
        min_duration=0.1,
        max_duration=5.0,
    )

    # 0.1s silence + 0.2s tone + 0.1s silence
    sr = 24000
    silence = torch.zeros(int(0.1 * sr))
    tone = torch.sin(torch.linspace(0, 2 * torch.pi * 440, int(0.2 * sr)))
    waveform = torch.cat([silence, tone, silence])

    processed, is_valid = pre.preprocess(waveform, sr)
    assert is_valid
    assert processed.shape[0] == 1
    assert processed.shape[1] < waveform.shape[0]

    rms = processed.pow(2).mean().sqrt().item()
    target_rms = 10 ** (-20.0 / 20.0)
    assert abs(rms - target_rms) < 0.05


def test_collator_invalid_audio_fallback(monkeypatch):
    from harmonica.data.collate import Collator
    from harmonica.text import CharTokenizer

    class MockCodec:
        sample_rate = 24000
        n_codebooks = 8

        def encode(self, audio):
            return torch.randint(0, 1024, (1, 8, 10))

    # Short waveform to force invalid
    def fake_load(_):
        return torch.zeros(1, 1000), 24000

    monkeypatch.setattr("torchaudio.load", lambda path: fake_load(path))

    pre = AudioPreprocessor(min_duration=1.0, max_duration=2.0, target_sample_rate=24000)
    collator = Collator(tokenizer=CharTokenizer(), codec=MockCodec(), preprocessor=pre)

    samples = [{"text": "hi", "audio_path": "/tmp/fake.wav", "idx": 0}]
    batch = collator(samples)

    assert batch.codec_tokens.shape[0] == 1
    assert batch.codec_tokens.shape[1] == 8
    assert batch.codec_tokens.shape[2] >= 1


def test_collator_handles_streaming_audio():
    from harmonica.data.collate import Collator
    from harmonica.text import CharTokenizer

    class MockCodec:
        sample_rate = 24000
        n_codebooks = 8

        def encode(self, audio):
            return torch.randint(0, 1024, (1, 8, 10))

    collator = Collator(tokenizer=CharTokenizer(), codec=MockCodec())
    samples = [{
        "text": "hello",
        "audio_waveform": torch.randn(1, 24000),
        "audio_sample_rate": 24000,
        "idx": 0,
    }]

    batch = collator(samples)
    assert batch.codec_tokens.shape[0] == 1
    assert batch.codec_tokens.shape[1] == 8


def test_checkpoint_restores_rng_state(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(1234)
    first = torch.rand(1).item()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=0,
        config={},
        path=str(ckpt_path),
        scaler=None,
        metrics={},
        random_state=None,
        sampler=None,
        training_time=0.0,
    )

    # Expected next value after restoring saved state
    expected_next = torch.rand(1).item()
    _ = torch.rand(1).item()

    # Restore RNG state from checkpoint
    load_checkpoint(str(ckpt_path), model=model, optimizer=optimizer)
    restored_next = torch.rand(1).item()

    assert restored_next == pytest.approx(expected_next)


def test_teacher_forcing_ratio_schedule():
    dummy_model = torch.nn.Linear(4, 4)
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "max_steps": 100,
            "scheduled_sampling": True,
            "teacher_forcing_schedule": "linear",
            "teacher_forcing_start": 1.0,
            "teacher_forcing_end": 0.5,
        },
        "experiment": {},
        "device": {"prefer": "cpu"},
    }

    trainer = Trainer(
        model=dummy_model,
        config=config,
        train_loader=dummy_loader,
        device=torch.device("cpu"),
    )

    ratio_start = trainer.get_teacher_forcing_ratio(0, 100)
    ratio_mid = trainer.get_teacher_forcing_ratio(50, 100)
    ratio_end = trainer.get_teacher_forcing_ratio(100, 100)

    assert ratio_start == pytest.approx(1.0)
    assert ratio_mid < ratio_start
    assert ratio_end == pytest.approx(0.5)


def _make_small_ar_model(length_control_mode: str = "duration_predictor"):
    from harmonica.model import ARTransformer

    return ARTransformer(
        vocab_size=16,
        text_vocab_size=16,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
        dropout=0.0,
        max_seq_len=32,
        max_text_len=16,
        length_control_mode=length_control_mode,
        duration_hidden_dim=8,
    )


def test_duration_loss_metric_present():
    model = _make_small_ar_model(length_control_mode="duration_predictor")
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "max_steps": 10,
            "scheduled_sampling": False,
            "duration_loss_weight": 0.5,
        },
        "experiment": {},
        "device": {"prefer": "cpu"},
    }
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dummy_loader,
        device=torch.device("cpu"),
    )

    audio_tokens = torch.randint(0, 16, (2, 6))
    text_tokens = torch.randint(0, 16, (2, 4))
    text_lengths = torch.tensor([4, 3])
    audio_lengths = torch.tensor([6, 5])

    loss, metrics = trainer._ar_teacher_forcing_step(
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        prompt_tokens=None,
        audio_lengths=audio_lengths,
    )

    assert loss.item() > 0
    assert "duration_loss" in metrics


def test_stop_token_training_appends_target():
    model = _make_small_ar_model(length_control_mode="stop_token")
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "max_steps": 10,
            "scheduled_sampling": False,
        },
        "experiment": {},
        "device": {"prefer": "cpu"},
    }
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dummy_loader,
        device=torch.device("cpu"),
    )

    audio_tokens = torch.randint(0, 16, (2, 5))
    text_tokens = torch.randint(0, 16, (2, 4))
    text_lengths = torch.tensor([4, 4])
    audio_lengths = torch.tensor([5, 5])

    loss, metrics = trainer._ar_teacher_forcing_step(
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        prompt_tokens=None,
        audio_lengths=audio_lengths,
    )

    assert model.stop_token_id is not None
    assert model.vocab_size == 17
    assert trainer._last_pred_tokens.shape[1] == audio_tokens.shape[1] + 1
    assert loss.item() > 0
