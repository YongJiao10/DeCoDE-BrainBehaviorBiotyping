"""DeCoDE input loading and participant-level score aggregation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class DeCoDEData:
    participant_ids: np.ndarray
    background_fc: torch.Tensor
    target_fc: torch.Tensor
    behavior: np.ndarray
    behavior_names: np.ndarray


class BalanceSampler:
    """Shuffle and partition a dataset into a fixed number of balanced batches."""

    def __init__(self, data_source, n_steps: int):
        self.total_size = len(data_source)
        base_batch_size, remainder = divmod(self.total_size, n_steps)
        self.batch_sizes = [base_batch_size] * n_steps
        for index in range(remainder):
            self.batch_sizes[index] += 1

    def __iter__(self):
        indices = torch.randperm(self.total_size).tolist()
        start = 0
        for batch_size in self.batch_sizes:
            stop = start + batch_size
            yield indices[start:stop]
            start = stop

    def __len__(self):
        return len(self.batch_sizes)


def participant_average_scores(
    participant_ids: np.ndarray,
    *score_matrices: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Average one or more run-level score matrices within participant."""
    participant_ids = np.asarray(participant_ids).astype(str)
    unique_ids, inverse = np.unique(participant_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    averaged_matrices = []
    for scores in score_matrices:
        scores = np.asarray(scores, dtype=float)
        averaged = np.zeros((unique_ids.size, scores.shape[1]), dtype=float)
        np.add.at(averaged, inverse, scores)
        averaged /= counts[:, None]
        averaged_matrices.append(averaged)
    return unique_ids, tuple(averaged_matrices)


def columnwise_pearson(
    fc_scores: np.ndarray,
    behavior_scores: np.ndarray,
) -> np.ndarray:
    """Compute one Pearson correlation for each aligned canonical dimension."""
    fc_scores = np.asarray(fc_scores, dtype=float)
    behavior_scores = np.asarray(behavior_scores, dtype=float)
    fc_centered = fc_scores - fc_scores.mean(axis=0, keepdims=True)
    behavior_centered = behavior_scores - behavior_scores.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(fc_centered**2, axis=0)
        * np.sum(behavior_centered**2, axis=0)
    )
    if np.any(denominator <= np.finfo(float).eps):
        raise ValueError("A canonical score has zero variance")
    return np.sum(fc_centered * behavior_centered, axis=0) / denominator


def load_data(connectivity_file: Path, behavior_file: Path) -> DeCoDEData:
    """Load aligned target/background FC features and target behavior data."""
    with np.load(connectivity_file, allow_pickle=False) as archive:
        participant_ids = np.asarray(archive["participant_ids"]).astype(str)
        target_fc = np.asarray(archive["target_fc"], dtype=np.float32)
        background_fc = np.asarray(archive["background_fc"], dtype=np.float32)

    behavior_frame = pd.read_csv(
        behavior_file,
        dtype={"participant_id": str},
    ).set_index("participant_id")
    if behavior_frame.index.duplicated().any():
        raise ValueError("Behavior input contains duplicate participant_id rows")
    behavior = behavior_frame.loc[participant_ids].to_numpy(dtype=np.float32)

    if len(participant_ids) != len(target_fc) or len(target_fc) != len(behavior):
        raise ValueError("Participant IDs, target FC, and behavior rows are misaligned")
    if not np.isfinite(target_fc).all() or not np.isfinite(background_fc).all():
        raise ValueError("FC inputs contain non-finite values")
    if not np.isfinite(behavior).all():
        raise ValueError("Behavior inputs contain non-finite values")

    return DeCoDEData(
        participant_ids=participant_ids,
        background_fc=torch.from_numpy(background_fc),
        target_fc=torch.from_numpy(target_fc),
        behavior=behavior,
        behavior_names=behavior_frame.columns.to_numpy(dtype=str),
    )
