import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNoiseScheduler:
    """Add noise to latent codes according to a linear schedule."""
    
    def __init__(self, T = 1000, beta_start = 1e-4, beta_end = 0.02, device = 'cpu'):
        self.T = T
        self.device = device
        
        # Linear Schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars  = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, z0, t, noise = None):
        """Forward process: (q(z_t | z_0) = N(sqrt(alpha_bar_t) * z_0, (1-alpha_bar_t) * I)
        Args: 
            z0 = the latent codes without noise, shape (N, C)
            t = the timestep indices, shape (N,)
            noise = optional pre-sampled noise
        
        Returns:
            zt = the noisy latent codes, shape (N, C)
            noise = the noise that was added
        
        """  
        
        if noise is None:
            noise = torch.randn_like(z0)
            
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1) # (N, 1)
        z_t = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return z_t, noise
    
    @torch.no_grad()
    def denoise(self, zt, denoiser, start_t = None):
        """ a full denoising reseve process from z_T to z_0

        Args:
            zt: the noisy latent, shape (N, C)
            denoiser: the denoising model
            start_t: start from this timestep (default: T-1)
        
        Returns:
            z0: the denoised latent, shape (N, C)
        """
        
        if start_t is None:
            start_t = self.T - 1
        
        # Start from the noisy latent z_T
        x = zt.clone()
        
        for t_val in reversed(range(start_t + 1)):
            # Create a tensor of the current timestep for the batch
            t = torch.full((x.shape[0], ), t_val, device=self.device, dtype=torch.long)
            # Predict the noise using the denoiser
            eps_pred = denoiser(x, t)
            alpha_t = self.alphas[t_val]
            alpha_bar_t = self.alpha_bars[t_val]
            beta_t = self.betas[t_val]
            
            # DDPM reverse step: predict x_{t-1} from x_t
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )
            # Add noise for all steps except the last one
            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean
                
        return x

    # Move the scheduler to a specific device
    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


class CosineNoiseScheduler(LinearNoiseScheduler):
    """Cosine noise schedule (Nichol & Dhariwal 2021).

    Destroys signal more slowly than the linear schedule at high t,
    which preserves anomaly-detection signal at the noise levels where
    the VLB / denoising-error scoring tends to be strongest.
    Inherits add_noise / denoise / to from LinearNoiseScheduler.
    """

    def __init__(self, T=1000, s=0.008, device='cpu'):
        # Avoid super().__init__ since the linear schedule sets up its own betas
        self.T = T
        self.device = device

        steps = T + 1
        t = torch.linspace(0, T, steps, device=device) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bars = f / f[0]
        self.alpha_bars = alpha_bars[1:]  # length T

        # Recover beta_t from alpha_bars: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bars[:-1]])
        self.betas = (1.0 - self.alpha_bars / prev).clamp(min=1e-8, max=0.999)
        self.alphas = 1.0 - self.betas


class LDMAutoencoder(nn.Module):
    """Autoencoder with tunable KL weight (beta) for latent diffusion.

    When beta=0, this is a pure autoencoder (no regularization).
    When beta=1, this is a standard VAE.
    For LDM, we use beta that closes to 0, so the latent space preserves more signal
    while still being slightly regularized for stability.
    """

    def __init__(self, n_dims_data=1024, n_dims_code=64,
                 hidden_layer_sizes=[512, 256], beta=0.01):
        super().__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.beta = beta
        self.kwargs = dict(
            n_dims_data=n_dims_data, n_dims_code=n_dims_code,
            hidden_layer_sizes=hidden_layer_sizes, beta=beta)

        # encoder
        encoder_layers = []
        prev_dim = n_dims_data
        for h_dim in hidden_layer_sizes:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder_body = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, n_dims_code)
        self.fc_logvar = nn.Linear(prev_dim, n_dims_code)

        # Decoder
        decoder_layers = []
        decoder_hidden = list(reversed(hidden_layer_sizes))
        prev_dim = n_dims_code
        for h_dim in decoder_hidden:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, n_dims_data))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss(self, x):
        x_recon, mu, log_var = self.forward(x)
        N = x.shape[0]
        recon = F.mse_loss(x_recon, x, reduction='sum') / N
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / N
        total = recon + self.beta * kl
        return total, recon.item(), kl.item()

    def save_to_file(self, fpath):
        state_dict = self.state_dict()
        state_dict['kwargs'] = self.kwargs
        torch.save(state_dict, fpath)

    @classmethod
    def load_from_file(cls, fpath):
        state_dict = torch.load(fpath, weights_only=False)
        kwargs = state_dict.pop('kwargs')
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        half_dim = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        # Compute the sinusoidal embedding
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class NoisePredictionDenoiser(nn.Module):
    """ A MLP that predicts the noise given the noisy latent and the timestep embedding. """
    
    def __init__(self, latent_dim = 32, time_embed_dim = 64, hidden_dim = 256, n_hidden = 3):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(time_embed_dim)
        layers = []
        input_dim = latent_dim + time_embed_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))  # Output the predicted noise  
        self.mlp = nn.Sequential(*layers)

    def forward(self, zt, t):
        """
        Predict the noise given the noisy latent and the timestep.
        args:
            zt: the noisy latent, shape (N, C)
            t: the timestep indices, shape (N,)
        returns:
            eps_pred: the predicted noise, shape (N, C)
        """
        time_embed = self.time_embed(t)
        x = torch.cat([zt, time_embed], dim=-1)
        eps_pred = self.mlp(x)
        return eps_pred
        