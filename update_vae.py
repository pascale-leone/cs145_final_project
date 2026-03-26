import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            n_dims_code=2,
            n_dims_data=400,
            hidden_layer_sizes=[32]):
        super(VariationalAutoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        # q_sigma is gone — sigma is now learned per sample
        self.kwargs = dict(
            n_dims_code=n_dims_code,
            n_dims_data=n_dims_data,
            hidden_layer_sizes=hidden_layer_sizes)

        # --- Encoder body (shared hidden layers) ---
        # Stops one layer short of the code; the final projections are
        # separate heads for mu and log_var.
        encoder_layer_sizes = [n_dims_data] + hidden_layer_sizes
        self.n_hidden_layers = len(encoder_layer_sizes) - 1

        self.encoder_params = nn.ModuleList()
        self.encoder_activations = []
        for n_in, n_out in zip(encoder_layer_sizes[:-1], encoder_layer_sizes[1:]):
            self.encoder_params.append(nn.Linear(n_in, n_out))
            self.encoder_activations.append(F.relu)

        # Two separate linear heads: one for mu, one for log_var
        n_hidden_last = hidden_layer_sizes[-1] if hidden_layer_sizes else n_dims_data
        self.fc_mu      = nn.Linear(n_hidden_last, n_dims_code)
        self.fc_log_var = nn.Linear(n_hidden_last, n_dims_code)

        # --- Decoder (mirrors the full encoder including the code layer) ---
        decoder_layer_sizes = [n_dims_code] + list(reversed(hidden_layer_sizes)) + [n_dims_data]
        self.n_decoder_layers = len(decoder_layer_sizes) - 1

        self.decoder_params = nn.ModuleList()
        self.decoder_activations = []
        for n_in, n_out in zip(decoder_layer_sizes[:-1], decoder_layer_sizes[1:]):
            self.decoder_params.append(nn.Linear(n_in, n_out))
            self.decoder_activations.append(F.relu)
        self.decoder_activations[-1] = nn.Identity()

    def forward(self, x_ND):
        """Run the full VAE: encode → sample → decode.

        Returns
        -------
        xproba_ND : tensor, same shape as x_ND
            Bernoulli probabilities for each pixel.
        mu_NC : tensor, N x C
            Encoder mean (useful for visualisation / downstream tasks).
        log_var_NC : tensor, N x C
            Encoder log-variance (needed externally for the KL term).
        """
        mu_NC, log_var_NC = self.encode(x_ND)
        z_NC = self.draw_sample_from_q(mu_NC, log_var_NC)
        return self.decode(z_NC), mu_NC, log_var_NC

    def encode(self, x_ND):
        """Pass x through the shared hidden layers, then split into mu and log_var.

        Returns
        -------
        mu_NC : tensor, N x C
        log_var_NC : tensor, N x C
        """
        h = x_ND
        for linear, act in zip(self.encoder_params, self.encoder_activations):
            h = act(linear(h))
        mu_NC      = self.fc_mu(h)
        log_var_NC = self.fc_log_var(h)
        return mu_NC, log_var_NC

    def draw_sample_from_q(self, mu_NC, log_var_NC):
        """Reparameterisation trick: z = mu + std * epsilon, epsilon ~ N(0, I).

        During training, sigma is exp(0.5 * log_var) — fully learned and
        back-propagatable. At eval time, we return the mean directly for
        a deterministic, lower-variance reconstruction.

        Parameters
        ----------
        mu_NC : tensor, N x C
        log_var_NC : tensor, N x C

        Returns
        -------
        z_NC : tensor, N x C
        """
        if self.training:
            std_NC = torch.exp(0.5 * log_var_NC)   # sigma per sample, per dim
            eps_NC = torch.randn_like(std_NC)        # epsilon ~ N(0, I)
            return mu_NC + std_NC * eps_NC
        else:
            return mu_NC

    def decode(self, z_NC):
        h = z_NC
        for linear, act in zip(self.decoder_params, self.decoder_activations):
            h = act(linear(h))
        return h  # xproba_ND, values in (0, 1) via sigmoid

    def calc_vi_loss(self, x_ND, n_mc_samples=1, do_rescale=None):
        """Evidence lower bound (ELBO) loss = reconstruction + KL.

        KL term is the closed-form expression from Appendix B of Kingma &
        Welling (2013), now using the *learned* log_var instead of a fixed
        scalar sigma:

            KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

        Parameters
        ----------
        x_ND : tensor
        n_mc_samples : int
            Number of Monte-Carlo samples for the reconstruction term.
        do_rescale : any truthy value
            When set, divides the loss by N*D so that the scale stays
            comparable across batch sizes and input dimensions.

        Returns
        -------
        loss : scalar tensor
        sample_xproba_ND : tensor (last MC sample, for logging)
        """
        mu_NC, log_var_NC = self.encode(x_ND)

        N, D = x_ND.shape
        Pbatch = (N * D) if do_rescale else 1.0

        # Closed-form KL divergence: q(z | x) || p(z) = N(0, I)
        # Shape: scalar (summed over all N samples and C dimensions)
        kl = -0.5 * torch.sum(
            1 + log_var_NC - mu_NC.pow(2) - log_var_NC.exp()
        )

        total_loss = 0.0
        #beta = 10.0  # tune this — try 2, 4, 10
        for ss in range(n_mc_samples):
            z_NC           = self.draw_sample_from_q(mu_NC, log_var_NC)
            sample_xproba_ND = self.decode(z_NC)
            recon_loss = F.mse_loss(sample_xproba_ND, x_ND, reduction='mean')
            
            total_loss += recon_loss + kl
            #total_loss    += bce + kl
            print(f'  [mc {ss}] kl: {kl:.4f}  recon: {recon_loss:.4f}')
            #print(f'  [mc {ss}] kl: {kl:.2f}  bce: {bce:.2f}  total: {total_loss:.2f}')

        return total_loss / float(Pbatch * n_mc_samples), sample_xproba_ND

    def train_for_one_epoch_of_gradient_update_steps(
            self, optimizer, train_loader, device, epoch, args):
        """Perform one epoch of gradient updates.

        Steps through dataset one minibatch at a time, computing the ELBO
        loss and back-propagating at each step.
        """
        self.train()
        n_batch = len(train_loader)
        loss_per_batch = np.zeros(n_batch)
        num_batch_before_print = int(np.ceil(n_batch / 5))

        for batch_idx, (batch_data, _) in enumerate(train_loader):
            batch_x_ND = batch_data.to(device).view(-1, self.n_dims_data)

            optimizer.zero_grad()

            loss, batch_xproba_ND = self.calc_vi_loss(
                batch_x_ND,
                n_mc_samples=args.n_mc_samples,
                do_rescale=True)

            loss_per_batch[batch_idx] = loss.item()
            loss.backward()
            optimizer.step()

            is_last = (batch_idx + 1 == len(train_loader))
            if (batch_idx + 1) % num_batch_before_print == 0 or is_last:
                l1_dist = torch.mean(torch.abs(batch_x_ND - batch_xproba_ND))
                print(
                    "  epoch %3d | frac_seen %.2f | avg loss % .4f"
                    " | batch loss % .4f | batch l1 % .3f" % (
                        epoch,
                        (1 + batch_idx) / float(len(train_loader)),
                        loss_per_batch[:(1 + batch_idx)].mean(),
                        loss.item(),
                        l1_dist,
                    ))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_to_file(self, fpath):
        state_dict = self.state_dict()
        state_dict['kwargs'] = self.kwargs
        torch.save(state_dict, fpath)

    @classmethod
    def save_model_to_file(cls, model, fpath):
        model.save_to_file(fpath)

    @classmethod
    def load_model_from_file(cls, fpath):
        """Load a saved model.

        Usage
        -----
        >>> VariationalAutoencoder.load_model_from_file('path/to/model.pytorch')
        """
        state_dict = torch.load(fpath)
        kwargs = state_dict.pop('kwargs')
        model = cls(**kwargs)
        assert 'kwargs' not in state_dict
        model.load_state_dict(state_dict)
        return model