#!/usr/bin/env python3
"""DeCoDE brain-behavior demo on random matrices."""

import argparse

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset

import models


class BrainBehaviorDataset(Dataset):
    def __init__(self, target_fc, background_fc, behavior):
        self.target_fc = target_fc.astype(np.float32)
        self.background_fc = background_fc.astype(np.float32)
        self.behavior = behavior.astype(np.float32)

    def __len__(self):
        return max(self.target_fc.shape[0], self.background_fc.shape[0])

    def __getitem__(self, i):
        target_i = i % self.target_fc.shape[0]
        background_i = i % self.background_fc.shape[0]
        return self.target_fc[target_i], self.background_fc[background_i], self.behavior[target_i]

def main(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Random matrices stand in for target FC, background FC, and behavior inputs.
    target_fc = rng.normal(size=(args.n_target, args.n_fc)).astype(np.float32)
    background_fc = rng.normal(size=(args.n_background, args.n_fc)).astype(np.float32)
    behavior = rng.normal(size=(args.n_target, args.n_behavior)).astype(np.float32)
    if args.fc_demean:
        target_fc = (target_fc - target_fc.mean(axis=1, keepdims=True)).astype(np.float32)
        background_fc = (background_fc - background_fc.mean(axis=1, keepdims=True)).astype(np.float32)
    print(f"data: target_fc={target_fc.shape}, background_fc={background_fc.shape}, behavior={behavior.shape}")

    behavior = ((behavior - behavior.mean(0)) / behavior.std(0)).astype(np.float32)

    model = models.cVAE(args.n_fc, args.hidden_dims, args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset = BrainBehaviorDataset(target_fc, background_fc, behavior)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Train cVAE while aligning target-specific embeddings with behavior by DGCCA.
    for epoch in range(args.epochs):
        model.train()
        for x_tg, x_bg, y in loader:
            x_tg, x_bg, y = x_tg.to(device), x_bg.to(device), y.to(device)
            opt.zero_grad()
            bg_rec, tg_rec = model(x_bg, x_tg)
            rec_loss, kl_loss = model.loss(x_bg, x_tg, bg_rec, tg_rec)
            gcca_loss = models.GCCA([model.tg_spe_z, y])
            loss = rec_loss + args.beta * kl_loss + args.alpha * gcca_loss
            if args.gamma > 0:
                tc_loss, disc_loss = model.disentangle(model.tg_sha_z, model.tg_spe_z)
                loss = loss + args.gamma * tc_loss + disc_loss
            loss.backward()
            opt.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch + 1}/{args.epochs}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = torch.cat([
            model.encode(batch.to(device), model.encoder)[0].cpu()
            for batch in DataLoader(torch.from_numpy(target_fc), batch_size=args.batch_size)
        ]).to(device)
    y = torch.from_numpy(behavior).to(device)
    u = models.GCCA([z, y], return_U=True)
    fc_scores = models.project(z, u[0]).cpu().numpy()
    # Fit GMM on FC canonical scores to obtain biotype labels.
    gmm = GaussianMixture(args.n_biotypes, random_state=args.seed, n_init=20)
    biotypes = gmm.fit_predict(fc_scores)
    print(f"biotype counts: {np.bincount(biotypes, minlength=args.n_biotypes)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n-target", type=int, default=1000, help="Number of target samples.")
    p.add_argument("--n-background", type=int, default=1000, help="Number of background samples.")
    p.add_argument("--n-fc", type=int, default=4950, help="Number of FC features.")
    p.add_argument("--n-behavior", type=int, default=16, help="Number of behavioral measures.")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size. Must be > min(--latent-dim, --n-behavior)")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=256,
                   help="Hidden layer dimensions of the cVAE encoder/decoder.")
    p.add_argument("--latent-dim", type=int, default=32, help="Latent dimension for shared and specific embeddings.")
    p.add_argument("--alpha", type=float, default=15.0, help="DGCCA loss weight.")
    p.add_argument("--beta", type=float, default=1.0, help="KL loss weight.")
    p.add_argument("--gamma", type=float, default=5.0, help="Total-correlation loss weight.")
    p.add_argument("--lr", type=float, default=5e-6, help="Adam learning rate.")
    p.add_argument("--n-biotypes", type=int, default=3, help="Number of GMM biotypes.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--fc-demean", default=True, action=argparse.BooleanOptionalAction,
                   help="Apply sample-wise FC demeaning (use --no-fc-demean to disable).")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                   help="Use gpu for CUDA when available; otherwise run on CPU.")
    args = p.parse_args()
    main(args)
