import numpy as np
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.models.ddpm import LinearNoiseScheduler, NoisePredictionDenoiser, LDMAutoencoder

# random seed for reproducability
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# load the features extracted
with open("data/features/tad_rgb_features_32.pkl", "rb") as f:
    features = pickle.load(f)

video_keys = list(features.keys())
video_labels = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

# split into train / val / test sets (mirrors train_vae_tsn.py to avoid test-set leakage)
normal_keys = [k for k, l in zip(video_keys, video_labels) if l == 0]
abnormal_keys = [k for k, l in zip(video_keys, video_labels) if l == 1]

# 75% of normal -> train; remaining 25% split evenly into val/test
train_keys, normal_test_val_keys = train_test_split(normal_keys, test_size=0.25, random_state=42)
normal_val_keys, normal_test_keys = train_test_split(normal_test_val_keys, test_size=0.5, random_state=42)

# 50/50 split of abnormal into val/test
abnormal_val_keys, abnormal_test_keys = train_test_split(abnormal_keys, test_size=0.5, random_state=42)

val_keys = normal_val_keys + abnormal_val_keys
test_keys = normal_test_keys + abnormal_test_keys

print(f"Train videos (normal only): {len(train_keys)}")
print(f"Val   videos (normal / abnormal): {len(normal_val_keys)} / {len(abnormal_val_keys)}")
print(f"Test  videos (normal / abnormal): {len(normal_test_keys)} / {len(abnormal_test_keys)}")

# prepare training data
x_train_vids = np.stack([features[k] for k in train_keys])
x_val_vids   = np.stack([features[k] for k in val_keys])
x_test_vids  = np.stack([features[k] for k in test_keys])
y_val_vids   = np.array([0 if k in normal_val_keys  else 1 for k in val_keys])
y_test_vids  = np.array([0 if k in normal_test_keys else 1 for k in test_keys])

# normalize features using TRAIN stats only
x_train_flat_raw = x_train_vids.reshape(-1, x_train_vids.shape[-1])
feat_mean = np.mean(x_train_flat_raw, axis=0)
feat_std  = np.std(x_train_flat_raw, axis=0) + 1e-8

x_train_flat = (x_train_flat_raw - feat_mean) / feat_std
x_val_flat   = (x_val_vids.reshape(-1, x_val_vids.shape[-1])  - feat_mean) / feat_std
x_test_flat  = (x_test_vids.reshape(-1, x_test_vids.shape[-1]) - feat_mean) / feat_std

x_val_by_video  = x_val_flat.reshape(x_val_vids.shape)
x_test_by_video = x_test_flat.reshape(x_test_vids.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# train ldm autoencoder
print("\n" + "="*70)
print("Step 1: Training LDM Autoencoder (beta=0.01)")
print("="*70)

ldm_ae = LDMAutoencoder(
    n_dims_data=1024, n_dims_code=32,
    hidden_layer_sizes=[512, 256], beta=0.01
).to(device)

ae_optimizer = torch.optim.Adam(ldm_ae.parameters(), lr=1e-3)
ae_dataset = TensorDataset(torch.FloatTensor(x_train_flat), torch.zeros(len(x_train_flat)))
ae_loader = DataLoader(ae_dataset, batch_size=64, shuffle=True)

n_epochs_ae = 100
for epoch in range(1, n_epochs_ae + 1):
    ldm_ae.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n_b = 0
    for batch_x, _ in ae_loader:
        batch_x = batch_x.to(device)
        loss, recon, kl = ldm_ae.loss(batch_x)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
        total_recon += recon
        total_kl += kl
        n_b += 1
    if epoch % 20 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d} | loss {total_loss/n_b:.4f} | recon {total_recon/n_b:.4f} | kl {total_kl/n_b:.4f}")

ldm_ae.save_to_file("checkpoints/ldm_autoencoder_rgb.pt")
print("LDM Autoencoder saved to checkpoints/ldm_autoencoder_rgb.pt")

# extract the latents

print("\n" + "="*70)
print("Step 2: Extracting latent codes with LDM Autoencoder")
print("="*70)

ldm_ae.eval()
with torch.no_grad():
    x_tensor = torch.FloatTensor(x_train_flat).to(device)
    mu, _ = ldm_ae.encode(x_tensor)
    train_latents = mu.cpu()

print(f"Train latents shape: {train_latents.shape}")
print(f"Latent mean: {train_latents.mean():.4f}, std: {train_latents.std():.4f}")


# train diffusion model on latents
T = 1000
scheduler_latent = LinearNoiseScheduler(T = T, device=device)
denoiser_latent = NoisePredictionDenoiser(latent_dim= train_latents.shape[1], time_embed_dim=32, hidden_dim=256).to(device)
optimizer = torch.optim.Adam(denoiser_latent.parameters(), lr=1e-3)

train_dataset = TensorDataset(train_latents, torch.zeros(len(train_latents)))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

n_epochs_latent = 1000

print(f"Training latent diffusion model for {n_epochs_latent} epochs...")

for epoch in range(1, n_epochs_latent + 1):
    # train for one epoch
    denoiser_latent.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_latents, _ in train_loader:
        batch_latents = batch_latents.to(device)
        batch_size = batch_latents.shape[0]
        
        # sample random timesteps for each sample in the batch
        t = torch.randint(0, T, (batch_size,), device=device)
        
        # add noise to the latents according to the scheduler
        noisy_latents, noise = scheduler_latent.add_noise(batch_latents, t)
        
        # predict the noise using the denoiser
        noise_pred = denoiser_latent(noisy_latents, t)
        
        # compute loss (MSE between predicted noise and true noise)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{n_epochs_latent}, Loss: {total_loss / n_batches:.4f}")
torch.save(denoiser_latent.state_dict(), "checkpoints/latent_diffusion_model_rgb.pt")

def score_ldm_vlb(x_test_flat, encoder, scheduler, denoiser, device, n_timesteps = 20):
    """compute the variational lower bound error across multiple timesteps in the latent space"""
    denoiser.eval()
    video_scores = []

    # define a set of timesteps to evaluate the denoising error
    #timesteps = torch.linspace(0, scheduler.T - 1, n_timesteps, dtype= torch.long, device = device)
    timesteps = torch.arange(1, n_timesteps, device=device)  # dense, consecutive
    for vid_feats in x_test_flat:
        x = torch.FloatTensor(vid_feats).to(device)
        with torch.no_grad():
            mu, _ = encoder(x)
            total_error = torch.zeros(mu.shape[0], device=device)
            # for each timestep, add noise and compute the denoising error
            for i,t_val in enumerate(timesteps):
                if t_val == 0:
                    continue
                t = t_val.expand(mu.shape[0])

                t_prev = timesteps[i - 1] if i > 0 else torch.tensor(0, device=device)
                a_t, a_tm1, b_t, b_tm1 = scheduler.alpha_bars[t_val], scheduler.alpha_bars[t_prev], scheduler.betas[t_val], scheduler.betas[t_prev]

                
                beta_tilde = (1 - a_tm1) / (1 - a_t) * b_t
                weight = 1 / (2 * beta_tilde)

                N_samples = 5
                sample_errors = torch.zeros(mu.shape[0], device=device)
                for j in range(N_samples):
                    eps_true = torch.randn_like(mu)
                    z_t = torch.sqrt(a_t) * mu + torch.sqrt(1 - a_t) * eps_true

                    mu_tilde = (torch.sqrt(a_tm1)*b_t)/((1-a_t))*mu + (torch.sqrt(1-b_t)*(1-a_tm1))/((1-a_t))*z_t

                    eps = denoiser(z_t, t)
                    mu_theta = (1/torch.sqrt(1-b_t))*(z_t-(b_t/torch.sqrt(1-a_t)) * eps )

                    error = torch.mean((mu_tilde - mu_theta)**2, dim=1)  # shape (n_segments,)
                    sample_errors += error

                total_error += weight * (sample_errors/N_samples)
            seg_errors = total_error.cpu().numpy()  # no division by n_timesteps 
         
            video_scores.append(np.mean(np.sort(seg_errors)[-5:]))
    return np.array(video_scores)

# ---- Select best number of time steps on VALIDATION set ----
print("\nSelecting t* on validation set...")
n_timestep_grid = [100, 200, 300, 400, 500, 600, 700, 800]
val_aucs = {}
for t in n_timestep_grid:
    scores_val = score_ldm_vlb(x_val_by_video, ldm_ae.encode,
                                scheduler_latent, denoiser_latent, device,
                                n_timesteps=t)
    auc_val = roc_auc_score(y_val_vids, scores_val)
    val_aucs[t] = auc_val
    print(f"  val AUC @ t={t}: {auc_val:.4f}")

best_t = max(val_aucs, key=val_aucs.get)
print(f"Selected n_timesteps = {best_t} (val AUC = {val_aucs[best_t]:.4f})")

# report VLB error score
scores_vlb = score_ldm_vlb(x_test_by_video, ldm_ae.encode,
                                          scheduler_latent, denoiser_latent, device, n_timesteps=best_t)
auc_vlb = roc_auc_score(y_test_vids, scores_vlb)

print(f"  Latent diffusion (vlb error):      test AUC = {auc_vlb:.4f}")


# VLB-error ROC 
fpr_vlb, tpr_vlb, _ = roc_curve(y_test_vids, scores_vlb)

plt.figure(figsize=(7, 5))
plt.plot(fpr_vlb, tpr_vlb, color="darkorange",
         label=f"Latent Diffusion — VLB Error (AUC = {auc_vlb:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — LDM VLB Error on TAD")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/roc_curve_ldm_vlb_rgb.png", dpi=150)
plt.show()
print("Saved ROC curve to results/figures/roc_curve_ldm_vlb_rgb.png")

np.save("results/scores/ldm_test_scores.npy", scores_vlb) # save scores for calibration



