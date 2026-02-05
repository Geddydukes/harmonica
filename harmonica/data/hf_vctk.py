"""Hugging Face streaming dataset loader for VCTK."""

from typing import Optional, Dict, Any, Iterator

import torch
from torch.utils.data import IterableDataset


class HFVCTKStreamingDataset(IterableDataset):
    """Stream VCTK from Hugging Face datasets (text only in some configs)."""

    def __init__(
        self,
        dataset_name: str = "confit/vctk-full",
        split: str = "train",
        shuffle: bool = True,
        shuffle_buffer: int = 1000,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        try:
            import datasets  # type: ignore
        except Exception as exc:
            raise ImportError(
                "datasets is required for HF streaming. Install with `pip install datasets`."
            ) from exc

        ds = datasets.load_dataset(
            dataset_name, split=split, streaming=True, trust_remote_code=True
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        self._dataset = ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        dataset = self._dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        for sample in dataset:
            audio = sample.get("audio", None)
            text = sample.get("text", sample.get("sentence", ""))
            speaker_id = sample.get("speaker_id", sample.get("speaker", None))

            if audio is None or text is None:
                continue

            waveform = torch.tensor(audio["array"]).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            yield {
                "audio_waveform": waveform,
                "audio_sample_rate": audio["sampling_rate"],
                "text": text,
                "speaker_id": speaker_id,
            }


class HFVCTKDataset(torch.utils.data.Dataset):
    """Non-streaming HF VCTK loader that includes audio."""

    def __init__(
        self,
        dataset_name: str = "vctk",
        split: str = "train",
    ):
        try:
            import datasets  # type: ignore
        except Exception as exc:
            raise ImportError(
                "datasets is required for HF loading. Install with `pip install datasets`."
            ) from exc

        try:
            self._dataset = datasets.load_dataset(
                dataset_name, split=split, trust_remote_code=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HF dataset '{dataset_name}' with split '{split}'. "
                "If this dataset doesn't include audio, try a different dataset name "
                "(e.g., 'vctk') or use the local download path."
            ) from exc

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._dataset[idx]
        audio = sample.get("audio", None)
        text = sample.get("text", sample.get("sentence", ""))
        speaker_id = sample.get("speaker_id", sample.get("speaker", None))

        if audio is None:
            raise ValueError("Audio field missing from HF dataset sample")

        waveform = torch.tensor(audio["array"]).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return {
            "audio_waveform": waveform,
            "audio_sample_rate": audio["sampling_rate"],
            "text": text,
            "speaker_id": speaker_id,
        }
