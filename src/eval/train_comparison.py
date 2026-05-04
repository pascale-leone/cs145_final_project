import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.models.starter_vae import VariationalAutoencoder
from src.models.ddpm import LinearNoiseScheduler, NoisePredictionDenoiser as NoisePredictor

# ── 1. LOAD FEATURES ─────────────────────────────────────────────────────────

with open("data/features/tad_rgb_features_32.pkl", "rb") as f:
    features = pickle.load(f)

video_keys = list(features.keys())
video_labels = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

# ── 2. SPLIT AT VIDEO LEVEL (same split as baseline) ─────────────────────────

normal_keys   = [k for k, l in zip(video_keys, video_labels) if l == 0]
abnormal_keys = [k for k, l in zip(video_keys, video_labels) if l == 1]

train_keys, normal_test_keys = train_test_split(
    normal_keys, test_size=0.2, random_state=42
)
test_keys = normal_test_keys + abnormal_keys

print(f"Train videos (normal only): {len(train_keys)}")
print(f"Test videos (normal):       {len(normal_test_keys)}")
print(f"Test videos (abnormal):     {len(abnormal_keys)}")

# ── 3. BUILD ARRAYS ──────────────────────────────────────────────────────────

X_train_vids = np.stack([features[k] for k in train_keys])
X_test_vids  = np.stack([features[k] for k in test_keys])
y_test_vids  = np.array([0 if k in normal_test_keys else 1 for k in test_keys])

# ── 4. Z-SCORE NORMALIZE (same stats as baseline) ───────────────────────────

X_train_flat_raw = X_train_vids.reshape(-1, 1024)
feat_mean = X_train_flat_raw.mean(axis=0)
feat_std  = X_train_flat_raw.std(axis=0) + 1e-8

X_train_norm = (X_train_vids - feat_mean) / feat_std
X_test_norm  = (X_test_vids  - feat_mean) / feat_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 5. LOAD TRAINED VAE ─────────────────────────────────────────────────────

vae = VariationalAutoencoder.load_model_from_file("checkpoints/vae_baseline.pt")
vae = vae.to(device)
vae.eval()

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1: Latent Diffusion with reconstruction-based scoring
#   encode → add noise → full denoise → decode → MSE in 1024-d feature space
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("APPROACH 1: Latent Diffusion (reconstruction scoring in feature space)")
print("="*70)

# Extract latent codes
print("Extracting latent codes from trained VAE encoder...")
X_train_flat = X_train_norm.reshape(-1, 1024)
with torch.no_grad():
    x_tensor = torch.FloatTensor(X_train_flat).to(device)
    mu, _ = vae.encode(x_tensor)
    train_latents = mu.cpu()

print(f"Latent codes shape: {train_latents.shape}")

# Train diffusion on latents
T = 1000
scheduler_latent = LinearNoiseScheduler(T=T, device=device)
denoiser_latent = NoisePredictor(
    latent_dim=32, time_embed_dim=64, hidden_dim=256, n_hidden=4
).to(device)
optimizer = torch.optim.Adam(denoiser_latent.parameters(), lr=1e-3)

train_dataset = TensorDataset(train_latents, torch.zeros(len(train_latents)))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

n_epochs_latent = 300
print(f"Training latent diffusion ({n_epochs_latent} epochs)...")
for epoch in range(1, n_epochs_latent + 1):
    denoiser_latent.train()
    total_loss = 0.0
    n_batch = 0
    for z0_batch, _ in train_loader:
        z0 = z0_batch.to(device)
        t = torch.randint(0, T, (z0.shape[0],), device=device)
        zt, noise = scheduler_latent.add_noise(z0, t)
        noise_pred = denoiser_latent(zt, t)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    if epoch % 50 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d} | loss {total_loss/n_batch:.6f}")

torch.save(denoiser_latent.state_dict(), "checkpoints/ldm_denoiser.pt")


def score_approach1_recon(X_test_norm, vae, denoiser, scheduler, device, noise_level=500):
    """Score by: encode → corrupt at timestep → full denoise → decode → MSE vs original."""
    denoiser.eval()
    video_scores = []
    for vid_feats in X_test_norm:
        x = torch.FloatTensor(vid_feats).to(device)  # (32, 1024)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            # Add noise at a specific level, then fully denoise
            t = torch.full((mu.shape[0],), noise_level, device=device, dtype=torch.long)
            zt, _ = scheduler.add_noise(mu, t)
            z_denoised = scheduler.denoise(zt, denoiser, start_t=noise_level)
            # Decode back to feature space
            x_recon = vae.decode(z_denoised)
        seg_errors = torch.mean((x - x_recon) ** 2, dim=1).cpu().numpy()
        video_scores.append(seg_errors.max())
    return np.array(video_scores)


def score_approach1_denoise(X_test_norm, vae, denoiser, scheduler, device, n_timesteps=20):
    """Score by: denoising error across multiple timesteps in latent space."""
    denoiser.eval()
    timesteps = torch.linspace(0, scheduler.T - 1, n_timesteps, dtype=torch.long, device=device)
    video_scores = []
    for vid_feats in X_test_norm:
        x = torch.FloatTensor(vid_feats).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            total_error = torch.zeros(mu.shape[0], device=device)
            for t_val in timesteps:
                t = t_val.expand(mu.shape[0])
                noise = torch.randn_like(mu)
                zt, _ = scheduler.add_noise(mu, t, noise=noise)
                noise_pred = denoiser(zt, t)
                error = torch.mean((noise - noise_pred) ** 2, dim=1)
                total_error += error
        seg_errors = (total_error / n_timesteps).cpu().numpy()
        video_scores.append(seg_errors.max())
    return np.array(video_scores)


# Score with reconstruction approach (multiple noise levels)
print("\nScoring Approach 1 variants...")
results = {}

for noise_level in [100, 250, 500, 750]:
    scores = score_approach1_recon(X_test_norm, vae, denoiser_latent, scheduler_latent, device, noise_level=noise_level)
    auc = roc_auc_score(y_test_vids, scores)
    results[f"A1-recon-t{noise_level}"] = auc
    print(f"  Approach 1 (recon, t={noise_level}): AUC = {auc:.4f}")

# Score with denoising error
scores_denoise = score_approach1_denoise(X_test_norm, vae, denoiser_latent, scheduler_latent, device)
auc_denoise = roc_auc_score(y_test_vids, scores_denoise)
results["A1-denoise-error"] = auc_denoise
print(f"  Approach 1 (denoise error):   AUC = {auc_denoise:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2: Direct Diffusion on 1024-d features (no VAE)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("APPROACH 2: Direct Diffusion on 1024-d features")
print("="*70)

T2 = 1000
scheduler_direct = LinearNoiseScheduler(T=T2, device=device)
denoiser_direct = NoisePredictor(
    latent_dim=1024, time_embed_dim=128, hidden_dim=512, n_hidden=4
).to(device)
optimizer2 = torch.optim.Adam(denoiser_direct.parameters(), lr=1e-3)

X_train_flat_tensor = torch.FloatTensor(X_train_norm.reshape(-1, 1024))
train_dataset2 = TensorDataset(X_train_flat_tensor, torch.zeros(len(X_train_flat_tensor)))
train_loader2 = DataLoader(train_dataset2, batch_size=128, shuffle=True)

n_epochs_direct = 200
print(f"Training direct diffusion on 1024-d features ({n_epochs_direct} epochs)...")
for epoch in range(1, n_epochs_direct + 1):
    denoiser_direct.train()
    total_loss = 0.0
    n_batch = 0
    for x0_batch, _ in train_loader2:
        x0 = x0_batch.to(device)
        t = torch.randint(0, T2, (x0.shape[0],), device=device)
        xt, noise = scheduler_direct.add_noise(x0, t)
        noise_pred = denoiser_direct(xt, t)
        loss = F.mse_loss(noise_pred, noise)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        total_loss += loss.item()
        n_batch += 1
    if epoch % 50 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d} | loss {total_loss/n_batch:.6f}")

torch.save(denoiser_direct.state_dict(), "checkpoints/direct_diffusion_denoiser.pt")


def score_direct_denoise(X_test_norm, denoiser, scheduler, device, n_timesteps=20):
    """Denoising error directly on 1024-d features."""
    denoiser.eval()
    timesteps = torch.linspace(0, scheduler.T - 1, n_timesteps, dtype=torch.long, device=device)
    video_scores = []
    for vid_feats in X_test_norm:
        x = torch.FloatTensor(vid_feats).to(device)
        with torch.no_grad():
            total_error = torch.zeros(x.shape[0], device=device)
            for t_val in timesteps:
                t = t_val.expand(x.shape[0])
                noise = torch.randn_like(x)
                xt, _ = scheduler.add_noise(x, t, noise=noise)
                noise_pred = denoiser(xt, t)
                error = torch.mean((noise - noise_pred) ** 2, dim=1)
                total_error += error
        seg_errors = (total_error / n_timesteps).cpu().numpy()
        video_scores.append(seg_errors.max())
    return np.array(video_scores)


def score_direct_recon(X_test_norm, denoiser, scheduler, device, noise_level=500):
    """Reconstruct via full denoising chain in 1024-d, compare to original."""
    denoiser.eval()
    video_scores = []
    for vid_feats in X_test_norm:
        x = torch.FloatTensor(vid_feats).to(device)
        with torch.no_grad():
            t = torch.full((x.shape[0],), noise_level, device=device, dtype=torch.long)
            xt, _ = scheduler.add_noise(x, t)
            x_denoised = scheduler.denoise(xt, denoiser, start_t=noise_level)
        seg_errors = torch.mean((x - x_denoised) ** 2, dim=1).cpu().numpy()
        video_scores.append(seg_errors.max())
    return np.array(video_scores)


print("\nScoring Approach 2 variants...")

# Denoising error
scores_d2 = score_direct_denoise(X_test_norm, denoiser_direct, scheduler_direct, device)
auc_d2 = roc_auc_score(y_test_vids, scores_d2)
results["A2-denoise-error"] = auc_d2
print(f"  Approach 2 (denoise error): AUC = {auc_d2:.4f}")

# Reconstruction at different noise levels
for noise_level in [100, 250, 500]:
    scores = score_direct_recon(X_test_norm, denoiser_direct, scheduler_direct, device, noise_level=noise_level)
    auc = roc_auc_score(y_test_vids, scores)
    results[f"A2-recon-t{noise_level}"] = auc
    print(f"  Approach 2 (recon, t={noise_level}): AUC = {auc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 3: Combined — VAE recon error + LDM denoising error
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("APPROACH 3: Combined VAE + LDM scoring")
print("="*70)

# Get baseline VAE scores
vae_scores = []
for vid_feats in X_test_norm:
    x = torch.FloatTensor(vid_feats).to(device)
    with torch.no_grad():
        x_recon, mu, log_var = vae(x)
    seg_errors = torch.mean((x - x_recon) ** 2, dim=1).cpu().numpy()
    vae_scores.append(seg_errors.max())
vae_scores = np.array(vae_scores)

auc_vae = roc_auc_score(y_test_vids, vae_scores)
results["VAE-baseline"] = auc_vae
print(f"  VAE baseline:     AUC = {auc_vae:.4f}")

# Normalize both score arrays to [0, 1] before combining
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

vae_norm = normalize(vae_scores)
ldm_denoise_norm = normalize(scores_denoise)
direct_denoise_norm = normalize(scores_d2)

for alpha in [0.3, 0.5, 0.7]:
    combined = alpha * vae_norm + (1 - alpha) * ldm_denoise_norm
    auc_c = roc_auc_score(y_test_vids, combined)
    results[f"A3-vae+latent-a{alpha}"] = auc_c
    print(f"  Combined VAE+Latent LDM (alpha={alpha}): AUC = {auc_c:.4f}")

for alpha in [0.3, 0.5, 0.7]:
    combined = alpha * vae_norm + (1 - alpha) * direct_denoise_norm
    auc_c = roc_auc_score(y_test_vids, combined)
    results[f"A3-vae+direct-a{alpha}"] = auc_c
    print(f"  Combined VAE+Direct DM (alpha={alpha}): AUC = {auc_c:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SUMMARY — All AUC-ROC Scores")
print("="*70)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    marker = " <-- BEST" if auc == max(results.values()) else ""
    print(f"  {name:35s} {auc:.4f}{marker}")

# ── FINAL: Best model detailed report ────────────────────────────────────────
best_name = max(results, key=results.get)
best_auc = results[best_name]
print(f"\nBest model: {best_name} (AUC = {best_auc:.4f})")

# ROC plot for best approaches
plt.figure(figsize=(8, 6))

# VAE baseline
fpr_v, tpr_v, _ = roc_curve(y_test_vids, vae_scores)
plt.plot(fpr_v, tpr_v, label=f"VAE baseline (AUC = {auc_vae:.4f})", linewidth=2)

# Best latent LDM
fpr_l, tpr_l, _ = roc_curve(y_test_vids, scores_denoise)
plt.plot(fpr_l, tpr_l, label=f"Latent LDM denoise (AUC = {auc_denoise:.4f})", linewidth=2)

# Direct diffusion
fpr_d, tpr_d, _ = roc_curve(y_test_vids, scores_d2)
plt.plot(fpr_d, tpr_d, label=f"Direct DM denoise (AUC = {auc_d2:.4f})", linewidth=2)

# Best combined if it's better
best_combined_name = max(
    [(k, v) for k, v in results.items() if k.startswith("A3")],
    key=lambda x: x[1]
)
if best_combined_name[1] > max(auc_vae, auc_denoise, auc_d2):
    # Recompute scores for the best combined
    alpha = float(best_combined_name[0].split("a")[1])
    if "direct" in best_combined_name[0]:
        combined_best = alpha * vae_norm + (1 - alpha) * direct_denoise_norm
    else:
        combined_best = alpha * vae_norm + (1 - alpha) * ldm_denoise_norm
    fpr_c, tpr_c, _ = roc_curve(y_test_vids, combined_best)
    plt.plot(fpr_c, tpr_c, label=f"Best combined (AUC = {best_combined_name[1]:.4f})", linewidth=2, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — VAE vs Diffusion Models on TAD")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("results/figures/roc_curve_comparison.png", dpi=150)
plt.show()
print("Saved ROC plot to results/figures/roc_curve_comparison.png")
