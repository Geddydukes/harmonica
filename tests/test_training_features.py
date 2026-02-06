"""Tests for training features and utilities."""

import torch
import pytest

from harmonica.utils.audio import AudioPreprocessor
from harmonica.training.checkpoint import save_checkpoint, load_checkpoint
from harmonica.training.trainer import Trainer, NARTrainer
from harmonica.model import NARTransformer


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return idx


class DummyBatch:
    def __init__(self):
        self.codec_tokens = torch.zeros(1, 1, 1, dtype=torch.long)
        self.audio_lengths = torch.ones(1, dtype=torch.long)
        self.text_tokens = torch.ones(1, 1, dtype=torch.long)
        self.text_lengths = torch.ones(1, dtype=torch.long)
        self.prompt_tokens = None
        self.prompt_lengths = None

    def to(self, _device):
        return self


class DummyBatchDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, _idx):
        return DummyBatch()


class DummyNARBatch:
    def __init__(self):
        # [B, K_total, L] where K_total = 1 (AR) + 2 (NAR targets)
        self.codec_tokens = torch.randint(0, 16, (1, 3, 6), dtype=torch.long)
        self.audio_lengths = torch.tensor([6], dtype=torch.long)
        self.text_tokens = torch.randint(0, 16, (1, 4), dtype=torch.long)
        self.text_lengths = torch.tensor([4], dtype=torch.long)
        self.prompt_tokens = None
        self.prompt_lengths = None

    def to(self, _device):
        return self


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


def test_update_step_conversion_from_legacy_micro_steps():
    dummy_model = torch.nn.Linear(4, 4)
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "grad_accum_steps": 4,
            "max_steps": 10,
            "warmup_steps": 5,
            "log_every": 8,
            "eval_every": 12,
            "checkpoint_every": 20,
            "warn_after_step": 6,
            "codebook_entropy_warmup_steps": 9,
            "token_noise_warmup_steps": 7,
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

    assert trainer.max_update_steps == 3
    assert trainer.warmup_update_steps == 2
    assert trainer.log_every_updates == 2
    assert trainer.eval_every_updates == 3
    assert trainer.checkpoint_every_updates == 5
    assert trainer.warn_after_updates == 2
    assert trainer.codebook_entropy_warmup_updates == 3
    assert trainer.token_noise_warmup_updates == 2


def test_entropy_and_noise_decay_use_update_steps():
    dummy_model = torch.nn.Linear(4, 4)
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "max_update_steps": 20,
            "warmup_update_steps": 0,
            "log_every_updates": 1,
            "eval_every_updates": 100,
            "checkpoint_every_updates": 100,
            "codebook_entropy_weight": 0.2,
            "codebook_entropy_warmup_updates": 4,
            "token_noise_prob": 0.3,
            "token_noise_warmup_updates": 4,
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

    assert trainer._codebook_entropy_weight_for_step(0) == pytest.approx(0.2)
    assert trainer._codebook_entropy_weight_for_step(2) == pytest.approx(0.1)
    assert trainer._codebook_entropy_weight_for_step(4) == pytest.approx(0.0)

    assert trainer._token_noise_prob_for_step(0) == pytest.approx(0.3)
    assert trainer._token_noise_prob_for_step(2) == pytest.approx(0.15)
    assert trainer._token_noise_prob_for_step(4) == pytest.approx(0.0)


def test_scheduled_sampling_uses_optimizer_step_not_micro_step():
    model = _make_small_ar_model(length_control_mode="duration_predictor")
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "scheduled_sampling": True,
            "teacher_forcing_schedule": "linear",
            "teacher_forcing_start": 1.0,
            "teacher_forcing_end": 0.5,
            "max_update_steps": 10,
            "warmup_update_steps": 0,
            "log_every_updates": 1,
            "eval_every_updates": 100,
            "checkpoint_every_updates": 100,
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
    trainer.optimizer_step = 5
    trainer.micro_step = 20
    trainer.global_step = trainer.micro_step

    called = {}

    def fake_ratio(step, max_steps):
        called["step"] = step
        called["max_steps"] = max_steps
        return 1.0

    trainer.get_teacher_forcing_ratio = fake_ratio

    audio_tokens = torch.randint(0, 16, (1, 4))
    text_tokens = torch.randint(0, 16, (1, 3))
    text_lengths = torch.tensor([3])
    audio_lengths = torch.tensor([4])

    loss, _ = trainer._ar_scheduled_sampling_step(
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        prompt_tokens=None,
        audio_lengths=audio_lengths,
    )

    assert loss.item() > 0
    assert called == {"step": 5, "max_steps": 10}


def test_logged_loss_not_over_scaled_by_grad_accum(tmp_path):
    class ConstantLossTrainer(Trainer):
        def _compute_loss(self, batch):  # type: ignore[override]
            param = next(self.model.parameters())
            loss = param.sum() * 0 + torch.tensor(2.0, device=param.device)
            return loss, {"accuracy": 0.0, "perplexity": torch.exp(loss.detach()).item()}

    model = torch.nn.Linear(2, 2)
    loader = torch.utils.data.DataLoader(
        DummyBatchDataset(),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
    )
    config = {
        "training": {
            "grad_accum_steps": 2,
            "max_update_steps": 1,
            "warmup_update_steps": 0,
            "log_every_updates": 1,
            "eval_every_updates": 100,
            "checkpoint_every_updates": 100,
            "mixed_precision": False,
        },
        "experiment": {
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "ckpts"),
        },
        "device": {"prefer": "cpu"},
    }

    trainer = ConstantLossTrainer(
        model=model,
        config=config,
        train_loader=loader,
        device=torch.device("cpu"),
    )
    summary = trainer.train(resume=False)

    assert summary["final_loss"] == pytest.approx(2.0, rel=1e-4)


def test_nar_training_step_uses_nar_text_encoder():
    model = NARTransformer(
        n_codebooks=2,
        vocab_size=16,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        max_seq_len=16,
        text_vocab_size=16,
        max_text_len=8,
        n_text_layers=1,
    )
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "max_update_steps": 2,
            "warmup_update_steps": 0,
            "log_every_updates": 1,
            "eval_every_updates": 10,
            "checkpoint_every_updates": 10,
            "mixed_precision": False,
        },
        "experiment": {},
        "device": {"prefer": "cpu"},
    }
    trainer = NARTrainer(
        model=model,
        config=config,
        train_loader=dummy_loader,
        device=torch.device("cpu"),
    )

    loss, metrics = trainer.nar_training_step(DummyNARBatch())

    assert loss.item() > 0
    assert "accuracy" in metrics
    assert trainer._last_pred_tokens is not None
    assert trainer._last_target_tokens is not None


def test_nar_training_step_applies_entropy_and_noise():
    model = NARTransformer(
        n_codebooks=2,
        vocab_size=16,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        max_seq_len=16,
        text_vocab_size=16,
        max_text_len=8,
        n_text_layers=1,
    )
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "max_update_steps": 2,
            "warmup_update_steps": 0,
            "log_every_updates": 1,
            "eval_every_updates": 10,
            "checkpoint_every_updates": 10,
            "mixed_precision": False,
            "codebook_entropy_weight": 0.2,
            "codebook_entropy_warmup_updates": 5,
            "codebook_usage_entropy_weight": 0.1,
            "codebook_usage_entropy_warmup_updates": 5,
            "token_noise_prob": 0.1,
            "token_noise_warmup_updates": 5,
            "nar_conditioning_noise_prob": 0.2,
            "nar_conditioning_noise_warmup_updates": 5,
        },
        "experiment": {},
        "device": {"prefer": "cpu"},
    }
    trainer = NARTrainer(
        model=model,
        config=config,
        train_loader=dummy_loader,
        device=torch.device("cpu"),
    )

    loss, metrics = trainer.nar_training_step(DummyNARBatch())

    assert loss.item() > 0
    assert "codebook_entropy_loss" in metrics
    assert "nar_entropy_weight" in metrics
    assert "nar_token_noise_prob" in metrics
    assert "codebook_usage_entropy_loss" in metrics
    assert "nar_usage_entropy_weight" in metrics
    assert "nar_conditioning_noise_prob" in metrics


def test_nar_trainer_passes_scheduled_teacher_forcing_ratio():
    model = NARTransformer(
        n_codebooks=2,
        vocab_size=16,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        max_seq_len=16,
        text_vocab_size=16,
        max_text_len=8,
        n_text_layers=1,
    )
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    config = {
        "training": {
            "max_update_steps": 10,
            "warmup_update_steps": 0,
            "log_every_updates": 1,
            "eval_every_updates": 10,
            "checkpoint_every_updates": 10,
            "mixed_precision": False,
            "nar_scheduled_sampling": True,
            "nar_teacher_forcing_schedule": "linear",
            "nar_teacher_forcing_start": 1.0,
            "nar_teacher_forcing_end": 0.0,
        },
        "experiment": {},
        "device": {"prefer": "cpu"},
    }
    trainer = NARTrainer(
        model=model,
        config=config,
        train_loader=dummy_loader,
        device=torch.device("cpu"),
    )
    trainer.optimizer_step = 5

    _, metrics = trainer.nar_training_step(DummyNARBatch())

    assert "nar_teacher_forcing_ratio" in metrics
    assert metrics["nar_teacher_forcing_ratio"] == pytest.approx(0.5, abs=1e-5)


def test_trainer_saves_and_loads_ema_checkpoint(tmp_path):
    class ConstantLossTrainer(Trainer):
        def _compute_loss(self, batch):  # type: ignore[override]
            param = next(self.model.parameters())
            loss = (param ** 2).mean()
            return loss, {"accuracy": 0.0, "perplexity": torch.exp(loss.detach()).item()}

    model = torch.nn.Linear(2, 2)
    loader = torch.utils.data.DataLoader(
        DummyBatchDataset(),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
    )
    config = {
        "training": {
            "grad_accum_steps": 1,
            "max_update_steps": 1,
            "warmup_update_steps": 0,
            "log_every_updates": 1,
            "eval_every_updates": 100,
            "checkpoint_every_updates": 100,
            "mixed_precision": False,
            "use_ema": True,
            "ema_decay": 0.9,
            "ema_update_every_updates": 1,
            "save_ema_in_checkpoints": True,
        },
        "experiment": {
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "ckpts"),
        },
        "device": {"prefer": "cpu"},
    }

    trainer = ConstantLossTrainer(
        model=model,
        config=config,
        train_loader=loader,
        device=torch.device("cpu"),
    )
    trainer.train(resume=False)
    ckpt_path = tmp_path / "ckpts" / "checkpoint_1.pt"

    assert ckpt_path.exists()

    loaded = load_checkpoint(str(ckpt_path), device=torch.device("cpu"))
    assert loaded.get("ema_state_dict") is not None
