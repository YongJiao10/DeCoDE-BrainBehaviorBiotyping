#!/usr/bin/env python3
"""Fit DeCoDE on all data using the best completed hyperparameter trial."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset

from data import BalanceSampler, load_data, participant_average_scores
from models import GCCA, cVAE


def main(args: argparse.Namespace) -> None:
    with args.best_trial_file.open(encoding="utf-8") as result_file:
        best_trial = json.load(result_file)
    parameters = best_trial["best_params"]
    epochs = int(best_trial["best_user_attrs"]["best_epoch"])
    canonical_dimensions = int(
        best_trial["best_user_attrs"]["canonical_dimensions"]
    )
    steps_per_epoch = int(best_trial["steps_per_epoch"])
    seed = int(best_trial["seed"])
    torch.manual_seed(seed)

    if args.device == "gpu" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("device: cpu (CUDA unavailable; fell back from gpu)")
    else:
        device = torch.device("cuda" if args.device == "gpu" else "cpu")
        print(f"device: {device}")

    data = load_data(args.connectivity_file, args.behavior_file)
    behavior_mean = data.behavior.mean(axis=0, keepdims=True)
    behavior_standard_deviation = data.behavior.std(axis=0, keepdims=True)
    behavior = torch.from_numpy(
        ((data.behavior - behavior_mean) / behavior_standard_deviation).astype(
            np.float32
        )
    )
    print(
        "data: "
        f"target_fc={tuple(data.target_fc.shape)}, "
        f"background_fc={tuple(data.background_fc.shape)}, "
        f"behavior={tuple(behavior.shape)}"
    )

    model = cVAE(
        in_dim=data.target_fc.shape[1],
        hidden_dim=int(parameters["hidden_dim"]),
        latent_dim=int(parameters["latent_dim"]),
        bias=False,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
    )
    background_loader = DataLoader(
        data.background_fc,
        batch_sampler=BalanceSampler(data.background_fc, steps_per_epoch),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    target_loader = DataLoader(
        TensorDataset(data.target_fc, behavior),
        batch_sampler=BalanceSampler(data.target_fc, steps_per_epoch),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    for epoch in range(epochs):
        model.train()
        for background, (target, batch_behavior) in zip(
            background_loader,
            target_loader,
        ):
            background = background.to(device)
            target = target.to(device)
            batch_behavior = batch_behavior.to(device)
            optimizer.zero_grad()
            background_reconstruction, target_reconstruction = model(
                background,
                target,
            )
            reconstruction_loss, kl_loss = model.loss(
                background,
                target,
                background_reconstruction,
                target_reconstruction,
            )
            loss = (
                reconstruction_loss
                + kl_loss
                + float(parameters["alpha"])
                * GCCA(
                    [model.tg_spe_z, batch_behavior],
                    top_K=canonical_dimensions,
                )
            )
            if float(parameters["gamma"]) > 0:
                total_correlation_loss, discriminator_loss = model.disentangle(
                    model.tg_sha_z,
                    model.tg_spe_z,
                )
                loss = (
                    loss
                    + float(parameters["gamma"]) * total_correlation_loss
                    + discriminator_loss
                )
            loss.backward()
            optimizer.step()
        if epoch == 0 or epoch + 1 == epochs:
            print(f"epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        latent = torch.cat(
            [
                model.encode(batch.to(device), model.encoder)[0].cpu()
                for batch in DataLoader(data.target_fc, batch_size=args.batch_size)
            ]
        ).to(device)
        behavior = behavior.to(device)
        loadings = GCCA(
            [latent, behavior],
            top_K=canonical_dimensions,
            return_U=True,
        )
        latent_mean = latent.mean(dim=0)
        behavior_center = behavior.mean(dim=0)
        run_fc_scores = torch.mm(latent - latent_mean, loadings[0]).cpu().numpy()
        run_behavior_scores = torch.mm(
            behavior - behavior_center,
            loadings[1],
        ).cpu().numpy()

    participant_ids, (participant_fc_scores, participant_behavior_scores) = (
        participant_average_scores(
            data.participant_ids,
            run_fc_scores,
            run_behavior_scores,
        )
    )
    gmm = GaussianMixture(args.n_biotypes, random_state=seed, n_init=20)
    biotypes = gmm.fit_predict(participant_fc_scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "parameters": parameters,
            "epochs": epochs,
            "canonical_dimensions": canonical_dimensions,
            "steps_per_epoch": steps_per_epoch,
            "seed": seed,
            "input_dim": int(data.target_fc.shape[1]),
            "bias": False,
        },
        args.output_dir / "model.pt",
    )
    np.savez(
        args.output_dir / "scores.npz",
        run_participant_ids=data.participant_ids,
        run_fc_scores=run_fc_scores,
        run_behavior_scores=run_behavior_scores,
        participant_ids=participant_ids,
        participant_fc_scores=participant_fc_scores,
        participant_behavior_scores=participant_behavior_scores,
        fc_loadings=loadings[0].cpu().numpy(),
        behavior_loadings=loadings[1].cpu().numpy(),
        latent_mean=latent_mean.cpu().numpy(),
        behavior_center=behavior_center.cpu().numpy(),
        behavior_mean=behavior_mean.ravel(),
        behavior_standard_deviation=behavior_standard_deviation.ravel(),
        behavior_names=data.behavior_names,
        gmm_weights=gmm.weights_,
        gmm_means=gmm.means_,
        gmm_covariances=gmm.covariances_,
        gmm_covariance_type=np.asarray(gmm.covariance_type),
        gmm_precisions_cholesky=gmm.precisions_cholesky_,
    )
    pd.DataFrame(
        {"participant_id": participant_ids, "biotype": biotypes}
    ).to_csv(args.output_dir / "biotypes.csv", index=False)
    summary = {
        "best_trial_file": str(args.best_trial_file),
        "study_name": best_trial["study_name"],
        "mode": best_trial["mode"],
        "best_value": best_trial["best_value"],
        "parameters": parameters,
        "epochs": epochs,
        "canonical_dimensions": canonical_dimensions,
        "steps_per_epoch": steps_per_epoch,
        "seed": seed,
        "n_target_runs": int(data.target_fc.shape[0]),
        "n_background_runs": int(data.background_fc.shape[0]),
        "n_participants": int(participant_ids.size),
        "n_biotypes": args.n_biotypes,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit DeCoDE on all data using a completed TPE or grid trial."
    )
    parser.add_argument("--connectivity_file", type=Path, required=True)
    parser.add_argument("--behavior_file", type=Path, required=True)
    parser.add_argument("--best_trial_file", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "final_fit",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_biotypes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    main(parser.parse_args())
