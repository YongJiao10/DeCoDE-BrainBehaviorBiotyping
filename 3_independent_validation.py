#!/usr/bin/env python3
"""Apply a fitted DeCoDE model to an independent cohort."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from data import columnwise_pearson, participant_average_scores
from models import cVAE


def main(args: argparse.Namespace) -> None:
    if args.device == "gpu" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("device: cpu (CUDA unavailable; fell back from gpu)")
    else:
        device = torch.device("cuda" if args.device == "gpu" else "cpu")
        print(f"device: {device}")

    checkpoint = torch.load(
        args.fit_dir / "model.pt",
        map_location=device,
        weights_only=True,
    )
    with np.load(args.fit_dir / "scores.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(args.connectivity_file, allow_pickle=False) as archive:
        run_participant_ids = np.asarray(archive["participant_ids"]).astype(str)
        fc = torch.from_numpy(np.asarray(archive["fc"], dtype=np.float32))

    behavior_names = reference["behavior_names"].astype(str).tolist()
    behavior_frame = pd.read_csv(
        args.behavior_file,
        dtype={"participant_id": str},
    ).set_index("participant_id")
    behavior = behavior_frame.loc[run_participant_ids, behavior_names].to_numpy(
        dtype=np.float32
    )
    behavior = (
        behavior - reference["behavior_mean"]
    ) / reference["behavior_standard_deviation"]

    parameters = checkpoint["parameters"]
    model = cVAE(
        in_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(parameters["hidden_dim"]),
        latent_dim=int(parameters["latent_dim"]),
        bias=bool(checkpoint["bias"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        latent = torch.cat(
            [
                model.encode(batch.to(device), model.encoder)[0].cpu()
                for batch in DataLoader(fc, batch_size=args.batch_size)
            ]
        ).numpy()

    run_fc_scores = (latent - reference["latent_mean"]) @ reference["fc_loadings"]
    run_behavior_scores = (
        behavior - reference["behavior_center"]
    ) @ reference["behavior_loadings"]
    participant_ids, (participant_fc_scores, participant_behavior_scores) = (
        participant_average_scores(
            run_participant_ids,
            run_fc_scores,
            run_behavior_scores,
        )
    )
    gmm = GaussianMixture(
        n_components=reference["gmm_weights"].size,
        covariance_type=str(reference["gmm_covariance_type"].item()),
    )
    gmm.weights_ = reference["gmm_weights"]
    gmm.means_ = reference["gmm_means"]
    gmm.covariances_ = reference["gmm_covariances"]
    gmm.precisions_cholesky_ = reference["gmm_precisions_cholesky"]
    biotypes = gmm.predict(participant_fc_scores)
    correlations = columnwise_pearson(
        participant_fc_scores,
        participant_behavior_scores,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "scores.npz",
        run_participant_ids=run_participant_ids,
        run_fc_scores=run_fc_scores,
        run_behavior_scores=run_behavior_scores,
        participant_ids=participant_ids,
        participant_fc_scores=participant_fc_scores,
        participant_behavior_scores=participant_behavior_scores,
    )
    pd.DataFrame(
        {"participant_id": participant_ids, "biotype": biotypes}
    ).to_csv(args.output_dir / "biotypes.csv", index=False)
    summary = {
        "fit_dir": str(args.fit_dir),
        "n_runs": int(fc.shape[0]),
        "n_participants": int(participant_ids.size),
        "canonical_dimensions": int(reference["fc_loadings"].shape[1]),
        "canonical_correlations": correlations.tolist(),
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a fitted DeCoDE model on an independent cohort."
    )
    parser.add_argument("--connectivity_file", type=Path, required=True)
    parser.add_argument("--behavior_file", type=Path, required=True)
    parser.add_argument("--fit_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "independent_validation",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    main(parser.parse_args())
