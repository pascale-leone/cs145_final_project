import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            n_dims_code=16,
            n_dims_data=1024,
            hidden_layer_sizes=[512, 256]):
        super(VariationalAutoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.kwargs = dict(
            n_dims_code=n_dims_code,
            n_dims_data=n_dims_data,
            hidden_layer_sizes=hidden_layer_sizes)

        # ── Encoder ──────────────────────────────────────────────────────
        encoder_layers = []
        prev_dim = n_dims_data
        for h_dim in hidden_layer_sizes:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder_body = nn.Sequential(*encoder_layers)

        # Two heads: one for mean, one for log-variance
        self.fc_mu = nn.Linear(prev_dim, n_dims_code)
        self.fc_logvar = nn.Linear(prev_dim, n_dims_code)

        # ── Decoder ──────────────────────────────────────────────────────
        decoder_layers = []
        decoder_hidden = list(reversed(hidden_layer_sizes))
        prev_dim = n_dims_code
        for h_dim in decoder_hidden:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, n_dims_data))
        # No final activation — Gaussian decoder for continuous features
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x_ND):
        """Encode input to latent distribution parameters (mu, log_var)."""
        h = self.encoder_body(x_ND)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, log_var):
        """Sample z from q(z|x) using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decode(self, z_NC):
        """Decode latent code to reconstruction."""
        return self.decoder(z_NC)

    def forward(self, x_ND):
        """Full forward pass: encode, sample, decode.

        Returns
        -------
        x_recon : reconstructed input
        mu      : encoder mean
        log_var : encoder log-variance
        """
        mu, log_var = self.encode(x_ND)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def calc_vi_loss(self, x_ND, n_mc_samples=1):
        """Compute the negative ELBO (reconstruction MSE + KL divergence).

        Reconstruction: MSE (Gaussian decoder with fixed unit variance)
        KL: Closed-form KL(q(z|x) || p(z)) for diagonal Gaussian q and N(0,I) prior
        """
        mu, log_var = self.encode(x_ND)
        N = x_ND.shape[0]

        total_recon = 0.0
        for _ in range(n_mc_samples):
            z = self.reparameterize(mu, log_var)
            x_recon = self.decode(z)
            # MSE reconstruction loss, summed over features, averaged over batch
            total_recon += F.mse_loss(x_recon, x_ND, reduction='sum') / N

        recon_loss = total_recon / n_mc_samples

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        # Averaged over batch
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / N

        loss = recon_loss + kl_loss
        return loss, x_recon, recon_loss.item(), kl_loss.item()

    def train_for_one_epoch(self, optimizer, train_loader, device, epoch):
        """Perform one epoch of gradient updates."""
        self.train()
        n_batch = len(train_loader)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch_idx, (batch_data, _) in enumerate(train_loader):
            x = batch_data.to(device)
            optimizer.zero_grad()
            loss, _, recon, kl = self.calc_vi_loss(x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon
            total_kl += kl

        avg_loss = total_loss / n_batch
        avg_recon = total_recon / n_batch
        avg_kl = total_kl / n_batch
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | loss {avg_loss:.4f} | recon {avg_recon:.4f} | kl {avg_kl:.4f}")
        return avg_loss

    def save_to_file(self, fpath):
        state_dict = self.state_dict()
        state_dict['kwargs'] = self.kwargs
        torch.save(state_dict, fpath)

    @classmethod
    def load_model_from_file(cls, fpath):
        state_dict = torch.load(fpath, weights_only=False)
        kwargs = state_dict.pop('kwargs')
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model
