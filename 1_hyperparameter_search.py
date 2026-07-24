#!/usr/bin/env python3
"""Cross-validated Optuna hyperparameter optimization for DeCoDE."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset

from data import (
    BalanceSampler,
    DeCoDEData,
    columnwise_pearson,
    load_data,
    participant_average_scores,
)
from models import GCCA, cVAE, project


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingSettings:
    max_epochs: int
    steps_per_epoch: int
    record_every_epochs: int
    early_stopping_start_epoch: int
    early_stopping_patience_epochs: int
    early_stopping_dimensions: int
    num_workers: int
    device: str
    seed: int


@dataclass(frozen=True)
class FoldRecord:
    """Participant-level held-out scores recorded for one CV fold."""

    participant_ids: np.ndarray
    epochs: np.ndarray
    fc_scores: tuple[np.ndarray, ...]
    behavior_scores: tuple[np.ndarray, ...]
    behavior_loadings: tuple[np.ndarray, ...]


def align_components(
    reference_loadings: np.ndarray,
    fold_loadings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fold-column order and signs matching a reference solution."""
    reference_loadings = np.asarray(reference_loadings, dtype=float)
    fold_loadings = np.asarray(fold_loadings, dtype=float)
    dimension_count = reference_loadings.shape[1]
    combined = np.column_stack([reference_loadings, fold_loadings])
    loading_correlations = np.corrcoef(combined, rowvar=False)
    cross_correlations = loading_correlations[
        :dimension_count,
        dimension_count:,
    ]
    reference_indices, fold_indices = linear_sum_assignment(
        -np.abs(cross_correlations)
    )
    order = np.empty(dimension_count, dtype=int)
    signs = np.ones(dimension_count, dtype=float)
    order[reference_indices] = fold_indices
    matched_correlations = cross_correlations[reference_indices, fold_indices]
    signs[reference_indices] = np.where(matched_correlations < 0, -1.0, 1.0)
    return order, signs


def cross_validated_correlations(
    fold_records: list[FoldRecord],
    recorded_epochs: np.ndarray,
) -> np.ndarray:
    """Align, concatenate, and correlate participant-level held-out scores."""
    correlations = []
    for epoch in recorded_epochs:
        selected_records = []
        for record in fold_records:
            indices = np.flatnonzero(record.epochs == epoch)
            if indices.size != 1:
                raise RuntimeError(f"Fold record is missing epoch {epoch}")
            index = int(indices[0])
            selected_records.append(
                (
                    record.participant_ids,
                    record.fc_scores[index],
                    record.behavior_scores[index],
                    record.behavior_loadings[index],
                )
            )

        reference_loadings = selected_records[0][3]
        all_participant_ids = []
        all_fc_scores = []
        all_behavior_scores = []
        for participant_ids, fc_scores, behavior_scores, loadings in selected_records:
            order, signs = align_components(reference_loadings, loadings)
            all_participant_ids.append(participant_ids)
            all_fc_scores.append(fc_scores[:, order] * signs)
            all_behavior_scores.append(behavior_scores[:, order] * signs)

        participant_ids = np.concatenate(all_participant_ids)
        if np.unique(participant_ids).size != participant_ids.size:
            raise RuntimeError("A participant appears in more than one validation fold")
        correlations.append(
            columnwise_pearson(
                np.concatenate(all_fc_scores, axis=0),
                np.concatenate(all_behavior_scores, axis=0),
            )
        )
    return np.asarray(correlations, dtype=float)


def participant_kfold(
    participant_ids: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_ids, inverse = np.unique(participant_ids, return_inverse=True)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_participants, validation_participants in splitter.split(unique_ids):
        train_indices = np.flatnonzero(np.isin(inverse, train_participants))
        validation_indices = np.flatnonzero(np.isin(inverse, validation_participants))
        splits.append((train_indices, validation_indices))
    return splits


class DeCoDEFold(L.LightningModule):
    def __init__(
        self,
        model: cVAE,
        data: DeCoDEData,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        parameters: dict,
        settings: TrainingSettings,
        canonical_dimensions: int,
    ):
        super().__init__()
        self.model = model
        self.background_fc = data.background_fc
        self.target_train = data.target_fc[train_indices]
        self.target_validation = data.target_fc[validation_indices]
        self.validation_run_participant_ids = data.participant_ids[
            validation_indices
        ]
        self.validation_participant_ids = np.unique(
            self.validation_run_participant_ids
        )
        behavior_train = data.behavior[train_indices]
        behavior_validation = data.behavior[validation_indices]
        mean = behavior_train.mean(axis=0, keepdims=True)
        standard_deviation = behavior_train.std(axis=0, keepdims=True)
        if np.any(standard_deviation == 0):
            raise ValueError("A behavior variable has zero variance in a training fold")
        self.behavior_train = torch.as_tensor(
            (behavior_train - mean) / standard_deviation,
            dtype=torch.float32,
        )
        self.behavior_validation = torch.as_tensor(
            (behavior_validation - mean) / standard_deviation,
            dtype=torch.float32,
        )
        self.learning_rate = float(parameters["learning_rate"])
        self.alpha = float(parameters["alpha"])
        self.gamma = float(parameters["gamma"])
        self.settings = settings
        self.canonical_dimensions = canonical_dimensions
        self.latest_objective = -math.inf
        self.recorded_epochs: list[int] = []
        self.recorded_fc_scores: list[np.ndarray] = []
        self.recorded_behavior_scores: list[np.ndarray] = []
        self.recorded_behavior_loadings: list[np.ndarray] = []

    def train_dataloader(self):
        target_dataset = TensorDataset(self.target_train, self.behavior_train)
        loader_options = {
            "num_workers": self.settings.num_workers,
            "persistent_workers": self.settings.num_workers > 0,
        }
        background_loader = DataLoader(
            self.background_fc,
            batch_sampler=BalanceSampler(
                self.background_fc,
                self.settings.steps_per_epoch,
            ),
            **loader_options,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_sampler=BalanceSampler(
                target_dataset,
                self.settings.steps_per_epoch,
            ),
            **loader_options,
        )
        return CombinedLoader(
            {"background": background_loader, "target": target_loader},
            mode="min_size",
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, _):
        self.model.train()
        background = batch["background"]
        target, behavior = batch["target"]
        background_reconstruction, target_reconstruction = self.model(
            background,
            target,
        )
        reconstruction_loss, kl_loss = self.model.loss(
            background,
            target,
            background_reconstruction,
            target_reconstruction,
        )
        gcca_loss = GCCA(
            [self.model.tg_spe_z, behavior],
            top_K=self.canonical_dimensions,
        )
        total_loss = reconstruction_loss + kl_loss + self.alpha * gcca_loss
        if self.gamma > 0:
            total_correlation_loss, discriminator_loss = self.model.disentangle(
                self.model.tg_sha_z,
                self.model.tg_spe_z,
            )
            total_loss += self.gamma * total_correlation_loss + discriminator_loss
        return total_loss

    @torch.no_grad()
    def on_train_epoch_end(self):
        self.model.eval()
        target_train = self.target_train.to(self.device)
        target_validation = self.target_validation.to(self.device)
        behavior_train = self.behavior_train.to(self.device)
        behavior_validation = self.behavior_validation.to(self.device)
        latent_train, _ = self.model.encode(target_train, self.model.encoder)
        latent_validation, _ = self.model.encode(
            target_validation,
            self.model.encoder,
        )
        loadings = GCCA(
            [latent_train, behavior_train],
            top_K=self.canonical_dimensions,
            return_U=True,
        )
        latent_train_center = latent_train.mean(dim=0, keepdim=True)
        behavior_train_center = behavior_train.mean(dim=0, keepdim=True)
        fc_scores = (
            project(
                latent_validation,
                loadings[0],
                latent_train_center,
            )
            .detach()
            .cpu()
            .numpy()
        )
        behavior_scores = project(
            behavior_validation,
            loadings[1],
            behavior_train_center,
        ).detach().cpu().numpy()
        _, participant_scores = participant_average_scores(
            self.validation_run_participant_ids,
            fc_scores,
            behavior_scores,
        )
        participant_fc_scores, participant_behavior_scores = participant_scores
        correlations_array = columnwise_pearson(
            participant_fc_scores,
            participant_behavior_scores,
        )
        self.latest_objective = float(
            np.mean(correlations_array[: self.settings.early_stopping_dimensions])
        )
        epoch = self.current_epoch + 1
        if epoch % self.settings.record_every_epochs == 0:
            self.recorded_epochs.append(epoch)
            self.recorded_fc_scores.append(participant_fc_scores)
            self.recorded_behavior_scores.append(participant_behavior_scores)
            self.recorded_behavior_loadings.append(
                loadings[1].detach().cpu().numpy()
            )


class ValidationEarlyStopping(L.Callback):
    def __init__(self, start_epoch: int, patience_epochs: int):
        self.start_epoch = start_epoch
        self.patience_epochs = patience_epochs
        self.best_score = -math.inf
        self.best_epoch = 0

    def on_train_epoch_start(self, trainer, module: DeCoDEFold):
        completed_epoch = trainer.current_epoch
        if completed_epoch < self.start_epoch:
            return
        score = module.latest_objective
        if np.isfinite(score) and score > self.best_score:
            self.best_score = score
            self.best_epoch = completed_epoch
        if (
            self.best_epoch > 0
            and completed_epoch - self.best_epoch >= self.patience_epochs
        ):
            trainer.should_stop = True


def train_fold(
    data: DeCoDEData,
    split: tuple[np.ndarray, np.ndarray],
    parameters: dict,
    settings: TrainingSettings,
    canonical_dimensions: int,
) -> FoldRecord:
    L.seed_everything(settings.seed, workers=True, verbose=False)
    model = cVAE(
        in_dim=data.target_fc.shape[1],
        hidden_dim=int(parameters["hidden_dim"]),
        latent_dim=int(parameters["latent_dim"]),
        bias=False,
    )
    module = DeCoDEFold(
        model,
        data,
        split[0],
        split[1],
        parameters,
        settings,
        canonical_dimensions,
    )
    trainer = L.Trainer(
        max_epochs=settings.max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        accelerator=settings.device,
        devices=1,
        num_sanity_val_steps=0,
        callbacks=[
            ValidationEarlyStopping(
                settings.early_stopping_start_epoch,
                settings.early_stopping_patience_epochs,
            )
        ],
    )
    trainer.fit(module)
    return FoldRecord(
        participant_ids=module.validation_participant_ids.copy(),
        epochs=np.asarray(module.recorded_epochs, dtype=int),
        fc_scores=tuple(module.recorded_fc_scores),
        behavior_scores=tuple(module.recorded_behavior_scores),
        behavior_loadings=tuple(module.recorded_behavior_loadings),
    )


class SearchObjective:
    def __init__(
        self,
        args: argparse.Namespace,
        config: dict,
        study_directory: Path,
    ):
        self.args = args
        self.config = config
        self.seed = int(config["seed"])
        self.data = load_data(args.connectivity_file, args.behavior_file)
        fold_count = int(config["cross_validation"]["n_folds"])
        self.splits = participant_kfold(
            self.data.participant_ids,
            fold_count,
            self.seed,
        )
        training = config["training"]
        self.settings = TrainingSettings(
            max_epochs=int(training["max_epochs"]),
            steps_per_epoch=int(training["steps_per_epoch"]),
            record_every_epochs=int(training["record_every_epochs"]),
            early_stopping_start_epoch=int(
                training["early_stopping_start_epoch"]
            ),
            early_stopping_patience_epochs=int(
                training["early_stopping_patience_epochs"]
            ),
            early_stopping_dimensions=int(
                config["objective"]["early_stopping_canonical_dimensions"]
            ),
            num_workers=args.num_workers,
            device=args.device,
            seed=self.seed,
        )
        self.objective_dimensions = int(
            config["objective"][f"{args.mode}_canonical_dimensions"]
        )
        self.max_canonical_dimensions = int(
            config["objective"]["max_canonical_dimensions"]
        )
        self.objective_name = (
            "mean participant-level cross-validated FC--behavior Pearson "
            f"correlation across the first {self.objective_dimensions} "
            "canonical dimensions"
        )
        self.metrics_directory = study_directory / "trial_metrics"
        self.metrics_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, trial: optuna.Trial) -> float:
        if self.args.mode == "grid":
            parameters = {
                name: trial.suggest_categorical(name, values)
                for name, values in self.config["grid"].items()
            }
        else:
            search = self.config["tpe"]
            parameters = {
                "hidden_dim": trial.suggest_categorical(
                    "hidden_dim",
                    search["hidden_dim"],
                ),
                "latent_dim": trial.suggest_categorical(
                    "latent_dim",
                    search["latent_dim"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    **search["learning_rate"],
                ),
                "alpha": trial.suggest_float("alpha", **search["alpha"]),
                "gamma": trial.suggest_float("gamma", **search["gamma"]),
            }
        canonical_dimensions = min(
            self.max_canonical_dimensions,
            self.data.behavior.shape[1],
            int(parameters["latent_dim"]),
        )
        # Sample one configuration and train it across participant-level folds.
        fold_records = [
            train_fold(
                self.data,
                split,
                parameters,
                self.settings,
                canonical_dimensions,
            )
            for split in self.splits
        ]
        # Align components, concatenate participant-level held-out scores,
        # then compute one cross-validated correlation per dimension.
        shared_epochs = set(fold_records[0].epochs.tolist())
        for record in fold_records[1:]:
            shared_epochs.intersection_update(record.epochs.tolist())
        recorded_epochs = np.asarray(sorted(shared_epochs), dtype=int)
        recorded_cross_validated_correlations = cross_validated_correlations(
            fold_records,
            recorded_epochs,
        )
        epoch_objectives = recorded_cross_validated_correlations[
            :, : self.objective_dimensions
        ].mean(axis=1)
        if not np.isfinite(epoch_objectives).all():
            raise RuntimeError("Non-finite validation correlation encountered")
        best_index = int(np.argmax(epoch_objectives))
        score = float(epoch_objectives[best_index])
        best_correlations = recorded_cross_validated_correlations[
            best_index, : self.objective_dimensions
        ]
        trial.set_user_attr("best_epoch", int(recorded_epochs[best_index]))
        trial.set_user_attr("canonical_dimensions", canonical_dimensions)
        for dimension, correlation in enumerate(best_correlations, start=1):
            trial.set_user_attr(
                f"validation_corr_dim{dimension}",
                float(correlation),
            )

        # Record trial settings, validation curves, and the best epoch.
        metrics = {
            "trial": trial.number,
            "parameters": parameters,
            "objective": self.objective_name,
            "best_epoch": int(recorded_epochs[best_index]),
            "canonical_dimensions": canonical_dimensions,
            "best_value": score,
            "best_dimension_correlations": best_correlations.tolist(),
            "recorded_epochs": recorded_epochs.tolist(),
            "recorded_cross_validated_correlations": (
                recorded_cross_validated_correlations.tolist()
            ),
            "seed": self.seed,
            "n_folds": len(self.splits),
            "steps_per_epoch": self.settings.steps_per_epoch,
        }
        metric_file = self.metrics_directory / f"trial_{trial.number:04d}.json"
        metric_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DeCoDE TPE or grid hyperparameter search."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode, description in (
        ("tpe", "Run an Optuna TPE search."),
        ("grid", "Run a full-factorial grid search."),
    ):
        command = subparsers.add_parser(mode, help=description)
        command.add_argument(
            "--config",
            type=Path,
            default=ROOT / "hyperparameter_config.json",
        )
        command.add_argument("--connectivity_file", type=Path, required=True)
        command.add_argument("--behavior_file", type=Path, required=True)
        command.add_argument(
            "--output_dir",
            type=Path,
            default=ROOT / "outputs/hyperparameter_optimization",
        )
        command.add_argument("--study_name")
        command.add_argument("--n_trials", type=int)
        command.add_argument("--num_workers", type=int, default=0)
        command.add_argument(
            "--device",
            choices=("cpu", "gpu"),
            default="gpu",
            help="Use CUDA GPU when available; otherwise fall back to CPU.",
        )
    # Load configuration, inputs, and the requested search mode.
    args = parser.parse_args()
    if args.device == "gpu" and not torch.cuda.is_available():
        args.device = "cpu"
        print("device: cpu (CUDA unavailable; fell back from gpu)")
    else:
        print(f"device: {args.device}")
    with args.config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if args.n_trials is not None:
        target_trials = args.n_trials
    elif args.mode == "tpe":
        target_trials = int(config["tpe"]["n_trials"])
    else:
        target_trials = math.prod(len(values) for values in config["grid"].values())
    study_name = args.study_name or f"decode_{args.mode}_search"
    study_directory = args.output_dir / study_name
    study_directory.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(study_directory / 'study.sqlite3').resolve()}"
    # Create or resume the Optuna study.
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=(
            optuna.samplers.TPESampler(seed=int(config["seed"]))
            if args.mode == "tpe"
            else optuna.samplers.GridSampler(
                config["grid"],
                seed=int(config["seed"]),
            )
        ),
        # Keep each trial's recorded trajectory for epoch selection.
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=True,
    )
    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining_trials = max(0, target_trials - completed_trials)
    objective = SearchObjective(args, config, study_directory)
    # Run the remaining TPE trials or grid configurations.
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
        )
    # Save the complete trial table and best-configuration summary.
    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    study.trials_dataframe().to_csv(study_directory / "trials.csv", index=False)
    best_trial = study.best_trial
    summary = {
        "study_name": study_name,
        "mode": args.mode,
        "objective": objective.objective_name,
        "best_value": study.best_value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "seed": int(config["seed"]),
        "steps_per_epoch": objective.settings.steps_per_epoch,
        "completed_trials": completed_trials,
    }
    (study_directory / "best_trial.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
