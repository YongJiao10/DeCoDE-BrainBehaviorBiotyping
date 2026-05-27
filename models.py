import torch
import torch.nn as nn
from typing import List, Optional

def build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int, *, bias: bool = False) -> nn.Sequential:
    """Builds a multi-layer perceptron with GELU activation."""
    layers: List[nn.Module] = []
    dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [int(output_dim)]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)

class BaseGaussianVAE(nn.Module):
    """Base class for Gaussian Variational Autoencoders."""
    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        return torch.sum((original - reconstructed).pow(2), dim=1).mean()

    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    def encode(self, x: torch.Tensor, encoder: Optional[nn.Module] = None):
        encoder = self.encoder if encoder is None else encoder
        out = encoder(x)
        mu, log_var = out.chunk(2, dim=1)
        return mu, log_var

class VAE(BaseGaussianVAE):
    """Standard Variational Autoencoder."""
    def __init__(self, in_dim: int, hidden_dim: int | List[int], latent_dim: int, bias: bool = False):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        self.encoder = build_mlp(in_dim, hidden_dim, latent_dim * 2, bias=bias)
        self.decoder = build_mlp(latent_dim, hidden_dim[::-1], in_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mu, self.log_var = self.encode(x)
        self.z = self.reparameterize(self.mu, self.log_var)
        return self.decoder(self.z)

class cVAE(VAE):
    """
    Contrastive VAE (cVAE) for disentangling salient features from background.
    Used as the backbone for DeCoDE.
    """
    def __init__(self, in_dim: int, hidden_dim: int | List[int], latent_dim: int, bias: bool = False):
        super().__init__(in_dim, hidden_dim, latent_dim, bias)
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        # Shared encoder for background features
        self.encoder_shared = build_mlp(in_dim, hidden_dim, latent_dim * 2, bias=bias)
        # Decoder takes concatenated (shared, specific) latents
        self.decoder = build_mlp(latent_dim * 2, hidden_dim[::-1], in_dim, bias=bias)

        # Discriminator for Total Correlation (TC) loss
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, bg: torch.Tensor, tg: torch.Tensor):
        self.bg_mu, self.bg_logvar = self.encode(bg, self.encoder_shared)
        self.tg_sha_mu, self.tg_sha_logvar = self.encode(tg, self.encoder_shared)
        self.tg_spe_mu, self.tg_spe_logvar = self.encode(tg, self.encoder)

        zeros = torch.zeros_like(self.bg_mu)

        if self.training:
            bg_z = self.reparameterize(self.bg_mu, self.bg_logvar)
            self.tg_sha_z = self.reparameterize(self.tg_sha_mu, self.tg_sha_logvar)
            self.tg_spe_z = self.reparameterize(self.tg_spe_mu, self.tg_spe_logvar)
            bg_recons = self.decoder(torch.cat([bg_z, zeros], dim=-1))
            tg_recons = self.decoder(torch.cat([self.tg_sha_z, self.tg_spe_z], dim=-1))
        else:
            bg_recons = self.decoder(torch.cat([self.bg_mu, zeros], dim=-1))
            tg_recons = self.decoder(torch.cat([self.tg_sha_mu, self.tg_spe_mu], dim=-1))

        return bg_recons, tg_recons

    def loss(self, bg: torch.Tensor, tg: torch.Tensor, bg_recons: torch.Tensor, tg_recons: torch.Tensor):
        reconstruction_loss = self.reconstruction_loss(bg, bg_recons) + self.reconstruction_loss(tg, tg_recons)
        kl_loss = (
            self.kl_loss(self.bg_mu, self.bg_logvar)
            + self.kl_loss(self.tg_sha_mu, self.tg_sha_logvar)
            + self.kl_loss(self.tg_spe_mu, self.tg_spe_logvar)
        )
        return reconstruction_loss, kl_loss

    def disentangle(self, tg_sha_z: torch.Tensor, tg_spe_z: torch.Tensor):
        """Computes Total Correlation (TC) loss using a discriminator."""
        batch_size = tg_sha_z.size(0)
        mid = batch_size - batch_size // 2
        z1, z2 = tg_sha_z[:mid], tg_sha_z[mid:]
        s1, s2 = tg_spe_z[: batch_size // 2], tg_spe_z[batch_size // 2 :]

        q_bar = torch.cat([torch.cat([s1, z2], dim=1), torch.cat([s2, z1], dim=1)], dim=0)
        q = torch.cat([tg_spe_z, tg_sha_z], dim=1)

        q_bar_score = self.discriminator(q_bar)
        q_score = self.discriminator(q)

        tc_loss = torch.log(q_score / (1 - q_score))
        discriminator_loss = -torch.log(q_score) - torch.log(1 - q_bar_score)
        return tc_loss.mean(), discriminator_loss.mean()

def GCCA(
    views: List[torch.Tensor],
    top_K: Optional[int] = None,
    *,
    return_U: bool = False,
):
    """Generalized Canonical Correlation Analysis (GCCA)."""
    top_K = min([H.shape[1] for H in views]) if top_K is None else int(top_K)
    eps = 1e-8
    at_list = []
    for view in views:
        hbar = view - view.mean(dim=0, keepdim=True)
        a, s, _ = hbar.svd(some=True, compute_uv=True)
        a = a[:, :top_K]
        s_thin = s[:top_K]
        s2_inv = 1.0 / (torch.mul(s_thin, s_thin) + eps)
        t2 = torch.mul(torch.mul(s_thin, s2_inv), s_thin)
        t2 = torch.where(t2 > eps, t2, torch.full_like(t2, eps))
        t = torch.diag(torch.sqrt(t2))
        at = torch.mm(a, t)
        at_list.append(at)

    m_tilde = torch.cat(at_list, dim=1)
    if return_U:
        u_list = []
        q, r = torch.linalg.qr(m_tilde, "reduced")
        v, _, _ = r.svd(some=False, compute_uv=True)
        g = q.mm(v[:, :top_K])
        for view in views:
            pinv = torch.linalg.pinv(view, rcond=eps)
            u_list.append(pinv.mm(g))
        for i in range(len(u_list)):
            u_list[i] = nn.Parameter(u_list[i].clone().detach(), requires_grad=False)
        return u_list

    _, s, _ = m_tilde.svd(some=True)
    s = s.topk(top_K)[0]
    corr = torch.sum(s)
    return -corr

def project(X: torch.Tensor, proj: torch.Tensor):
    """Projects data using the canonical correlation matrix."""
    canonical_score = X - X.mean(0, keepdims=True)
    return torch.mm(canonical_score, proj)

def get_corr(X: torch.Tensor, Y: torch.Tensor, U: List[torch.Tensor]):
    """Computes correlation between projected views."""
    score_X = project(X, U[0])
    score_Y = project(Y, U[1])
    score_X -= score_X.mean(0, keepdim=True)
    score_Y -= score_Y.mean(0, keepdim=True)
    std_product = torch.sqrt(torch.sum(score_X**2, dim=0) * torch.sum(score_Y**2, dim=0))
    corr = torch.sum(score_X * score_Y, dim=0) / std_product
    return corr
