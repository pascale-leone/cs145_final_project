import numpy as np
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ddpm import LinearNoiseScheduler, NoisePredictionDenoiser, LDMAutoencoder


# load the features extracted
with open("tad_twostream_features.pkl", "rb") as f:
    features = pickle.load(f)

video_keys = list(features.keys())
video_labels = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

# split into train / val / test sets (mirrors train_vae_tsn.py to avoid test-set leakage)
normal_keys = [k for k, l in zip(video_keys, video_labels) if l == 0]
abnormal_keys = [k for k, l in zip(video_keys, video_labels) if l == 1]

# 80% of normal -> train; remaining 20% split evenly into val/test
train_keys, normal_test_val_keys = train_test_split(normal_keys, test_size=0.2, random_state=42)
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
    n_dims_data=2048, n_dims_code=64,
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

ldm_ae.save_to_file("ldm_autoencoder.pt")
print("LDM Autoencoder saved to ldm_autoencoder.pt")

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
denoiser_latent = NoisePredictionDenoiser(latent_dim= train_latents.shape[1], time_embed_dim=64, hidden_dim=256).to(device)
optimizer = torch.optim.Adam(denoiser_latent.parameters(), lr=1e-3)

train_dataset = TensorDataset(train_latents, torch.zeros(len(train_latents)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

n_epochs_latent = 500

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
torch.save(denoiser_latent.state_dict(), "latent_diffusion_model.pt")

def score_ldm_recon(x_test_flat, encoder, decoder, scheduler, denoiser, device, noise_level = 500):
    """compute reconstruction error in feature space using the trained latent diffusion model"""
    denoiser.eval()
    video_scores = []
    for vid_feats in x_test_flat:
        with torch.no_grad():
            x = torch.FloatTensor(vid_feats).to(device)
            mu, _ = encoder(x)
            t = torch.full((mu.shape[0],), noise_level, dtype=torch.long, device=device)
            noisy_latents, _ = scheduler.add_noise(mu, t)
            latent_denoised = scheduler.denoise(noisy_latents, denoiser, start_t = noise_level)
            # decode the denoised latent back to feature space
            recon = decoder(latent_denoised)
        seg_errors = torch.mean ((x - recon) ** 2, dim=1).cpu().numpy()  # MSE per segment
        video_scores.append(seg_errors.max())  # take max error across segments as video score
    return np.array(video_scores)

def score_ldm_denoise_error(x_test_flat, encoder, scheduler, denoiser, device, n_timesteps = 20):
    """compute the denoising error across multiple timesteps in the latent space"""
    denoiser.eval()
    video_scores = []
    # define a set of timesteps to evaluate the denoising error
    timesteps = torch.linspace(0, scheduler.T - 1, n_timesteps, dtype= torch.long, device = device)
    for vid_feats in x_test_flat:
        x = torch.FloatTensor(vid_feats).to(device)
        with torch.no_grad():
            mu, _ = encoder(x)
            total_error = torch.zeros(mu.shape[0], device=device)
            # for each timestep, add noise and compute the denoising error
            for t_val in timesteps:
                t = t_val.expand(mu.shape[0])
                noise = torch.randn_like(mu)
                noisy_latents, _ = scheduler.add_noise(mu, t, noise = noise)
                noise_pred = denoiser(noisy_latents, t)
                error = torch.mean((noise_pred - noise) ** 2, dim=1)
                total_error += error
            seg_errors = (total_error/n_timesteps).cpu().numpy()  # average error across timesteps
            video_scores.append(seg_errors.max())  # take max error across segments as video score
    return np.array(video_scores)
                
# ---- Select best noise level on VALIDATION set ----
print("\nSelecting t* on validation set...")
noise_level_grid = [100, 250, 500, 750]
val_aucs = {}
for noise_level in noise_level_grid:
    scores_val = score_ldm_recon(x_val_by_video, ldm_ae.encode, ldm_ae.decode,
                                  scheduler_latent, denoiser_latent, device,
                                  noise_level=noise_level)
    auc_val = roc_auc_score(y_val_vids, scores_val)
    val_aucs[noise_level] = auc_val
    print(f"  val AUC @ t*={noise_level}: {auc_val:.4f}")

best_t = max(val_aucs, key=val_aucs.get)
print(f"Selected t* = {best_t} (val AUC = {val_aucs[best_t]:.4f})")

# ---- Evaluate on TEST set with selected t* ----
print("\nEvaluating on test set...")
results = {}
scores_test = score_ldm_recon(x_test_by_video, ldm_ae.encode, ldm_ae.decode,
                               scheduler_latent, denoiser_latent, device,
                               noise_level=best_t)
auc_test = roc_auc_score(y_test_vids, scores_test)
results[f"A1-recon-t{best_t}"] = auc_test
print(f"  Latent diffusion (recon, t*={best_t}): test AUC = {auc_test:.4f}")

# Also report denoise-error score (no hyperparameter to tune)
scores_denoise = score_ldm_denoise_error(x_test_by_video, ldm_ae.encode,
                                          scheduler_latent, denoiser_latent, device)
auc_denoise = roc_auc_score(y_test_vids, scores_denoise)
results["A1-denoise-error"] = auc_denoise
print(f"  Latent diffusion (denoise error):      test AUC = {auc_denoise:.4f}")

# Plot AUC results
labels = list(results.keys())
aucs = list(results.values())

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, aucs, color=["steelblue"] * 4 + ["darkorange"])
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline")
ax.set_ylim(0, 1)
ax.set_ylabel("AUC-ROC")
ax.set_title("Latent Diffusion Model — AUC by Scoring Method")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right")
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig("roc_auc_ldm.png", dpi=150)
plt.show()
print("Saved AUC bar chart to roc_auc_ldm.png")

#  Recon-error ROC (uses scores_test from the selected t*)
fpr_recon, tpr_recon, _ = roc_curve(y_test_vids, scores_test)
auc_recon = roc_auc_score(y_test_vids, scores_test)

plt.figure(figsize=(7, 5))
plt.plot(fpr_recon, tpr_recon, label=f"Latent Diffusion — Recon Error (AUC = {auc_recon:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — LDM Reconstruction Error (t* = {best_t}) on TAD")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_ldm_recon.png", dpi=150)
plt.show()
print("Saved ROC curve to roc_curve_ldm_recon.png")

# Denoise-error ROC (uses scores_denoise) 
fpr_denoise, tpr_denoise, _ = roc_curve(y_test_vids, scores_denoise)

plt.figure(figsize=(7, 5))
plt.plot(fpr_denoise, tpr_denoise, color="darkorange",
         label=f"Latent Diffusion — Denoise Error (AUC = {auc_denoise:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — LDM Denoising Error on TAD")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_ldm_denoise.png", dpi=150)
plt.show()
print("Saved ROC curve to roc_curve_ldm_denoise.png")

# Combined ROC plot
plt.figure(figsize=(7, 5))
plt.plot(fpr_recon,   tpr_recon,   label=f"Recon Error (AUC = {auc_recon:.4f})")
plt.plot(fpr_denoise, tpr_denoise, label=f"Denoise Error (AUC = {auc_denoise:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — LDM Scoring Methods on TAD")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_ldm_combined.png", dpi=150)
plt.show()
print("Saved combined ROC curve to roc_curve_ldm_combined.png")