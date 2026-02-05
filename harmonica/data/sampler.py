"""Custom samplers for curriculum learning and multi-dataset training."""

import random
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Sampler, Dataset, ConcatDataset


class CurriculumSampler(Sampler[int]):
    """Sampler that adjusts dataset weights during training.

    Used for curriculum learning where we gradually introduce
    harder examples or more diverse speakers.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[Dict[str, float]] = None,
        total_samples: int = 10000,
        seed: int = 42,
    ):
        """Initialize curriculum sampler.

        Args:
            datasets: List of datasets to sample from
            weights: Dict mapping dataset name to sampling weight
            total_samples: Number of samples per epoch
            seed: Random seed
        """
        self.datasets = datasets
        self.weights = weights or {}
        self.total_samples = total_samples
        self.seed = seed
        self.epoch = 0

        # Compute dataset boundaries in concatenated dataset
        self.boundaries = [0]
        for ds in datasets:
            self.boundaries.append(self.boundaries[-1] + len(ds))

        # Default weights if not specified
        if not self.weights:
            n = len(datasets)
            self.weights = {f"dataset_{i}": 1.0 / n for i in range(n)}

    def __iter__(self) -> Iterator[int]:
        """Generate sample indices."""
        rng = random.Random(self.seed + self.epoch)

        # Normalize weights
        weight_values = list(self.weights.values())
        total_weight = sum(weight_values)
        probs = [w / total_weight for w in weight_values]

        indices = []
        for _ in range(self.total_samples):
            # Sample dataset
            ds_idx = rng.choices(range(len(self.datasets)), weights=probs, k=1)[0]

            # Sample index within dataset
            ds_len = len(self.datasets[ds_idx])
            local_idx = rng.randint(0, ds_len - 1)

            # Convert to global index
            global_idx = self.boundaries[ds_idx] + local_idx
            indices.append(global_idx)

        return iter(indices)

    def __len__(self) -> int:
        return self.total_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducibility."""
        self.epoch = epoch

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update dataset weights."""
        self.weights = weights


class LengthBatchSampler(Sampler[List[int]]):
    """Sampler that groups samples by length for efficient batching.

    Minimizes padding by grouping similar-length samples together.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize length-based batch sampler.

        Args:
            lengths: List of sample lengths
            batch_size: Batch size
            drop_last: Drop last incomplete batch
            shuffle: Shuffle batches
            seed: Random seed
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices."""
        rng = random.Random(self.seed + self.epoch)

        # Create batches of similar lengths
        batches = []
        current_batch = []

        for idx in self.sorted_indices:
            current_batch.append(idx)
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        # Shuffle batches (but not within batches)
        if self.shuffle:
            rng.shuffle(batches)

        return iter(batches)

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducibility."""
        self.epoch = epoch


class BalancedSpeakerSampler(Sampler[int]):
    """Sampler that balances samples across speakers.

    Ensures each speaker is represented equally in each epoch,
    regardless of how many samples they have.
    """

    def __init__(
        self,
        speaker_ids: List[str],
        samples_per_speaker: int = 100,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize balanced speaker sampler.

        Args:
            speaker_ids: List of speaker IDs for each sample
            samples_per_speaker: Target samples per speaker per epoch
            shuffle: Shuffle samples
            seed: Random seed
        """
        self.speaker_ids = speaker_ids
        self.samples_per_speaker = samples_per_speaker
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group samples by speaker
        self.speaker_samples: Dict[str, List[int]] = {}
        for idx, speaker in enumerate(speaker_ids):
            if speaker not in self.speaker_samples:
                self.speaker_samples[speaker] = []
            self.speaker_samples[speaker].append(idx)

        self.speakers = list(self.speaker_samples.keys())

    def __iter__(self) -> Iterator[int]:
        """Generate sample indices."""
        rng = random.Random(self.seed + self.epoch)

        indices = []
        for speaker in self.speakers:
            speaker_indices = self.speaker_samples[speaker].copy()

            # Sample with replacement if needed
            if len(speaker_indices) < self.samples_per_speaker:
                sampled = rng.choices(speaker_indices, k=self.samples_per_speaker)
            else:
                sampled = rng.sample(speaker_indices, self.samples_per_speaker)

            indices.extend(sampled)

        if self.shuffle:
            rng.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.speakers) * self.samples_per_speaker

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducibility."""
        self.epoch = epoch
