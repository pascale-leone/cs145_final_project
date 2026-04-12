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
        