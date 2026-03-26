import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
#from starter_vae import VariationalAutoencoder
from update_vae import VariationalAutoencoder
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ── 1. LOAD FEATURES ─────────────────────────────────────────────────────────

with open("tad_rgb_features.pkl", "rb") as f:
    features = pickle.load(f)

video_keys = list(features.keys())
video_labels = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

# ── 2. SPLIT AT VIDEO LEVEL ───────────────────────────────────────────────────

normal_keys   = [k for k, l in zip(video_keys, video_labels) if l == 0]
abnormal_keys = [k for k, l in zip(video_keys, video_labels) if l == 1]

# Train on normal videos only — no abnormal videos ever seen during training
train_keys, normal_test_keys = train_test_split(
    normal_keys, test_size=0.2, random_state=42
)
test_keys = normal_test_keys + abnormal_keys

print(f"Train videos (normal only): {len(train_keys)}")
print(f"Test videos (normal):       {len(normal_test_keys)}")
print(f"Test videos (abnormal):     {len(abnormal_keys)}")

# ── 3. BUILD ARRAYS — KEEP (25, 1024) PER VIDEO ───────────────────────────────

# Stack into (N_videos, 25, 1024) — preserves temporal structure
X_train_vids = np.stack([features[k] for k in train_keys])   # (N_train, 25, 1024)
X_test_vids  = np.stack([features[k] for k in test_keys])    # (N_test,  25, 1024)
y_test_vids  = np.array([0 if k in normal_test_keys else 1 for k in test_keys])

# ── 4. NORMALIZE USING TRAIN STATS ONLY ──────────────────────────────────────

# X_min = X_train_vids.min()
# X_max = X_train_vids.max()

# X_train_norm = (X_train_vids - X_min) / (X_max - X_min)  # (N_train, 25, 1024)
# X_test_norm  = (X_test_vids  - X_min) / (X_max - X_min)  # (N_test,  25, 1024)

X_mean = X_train_vids.mean()
X_std  = X_train_vids.std()

X_train_norm = (X_train_vids - X_mean) / X_std
X_test_norm  = (X_test_vids  - X_mean) / X_std

normal_norms   = np.linalg.norm(X_test_norm[y_test_vids==0].reshape(len(normal_test_keys), -1), axis=1)
abnormal_norms = np.linalg.norm(X_test_norm[y_test_vids==1].reshape(len(abnormal_keys), -1), axis=1)
scores = np.concatenate([normal_norms, abnormal_norms])
print("Raw feature AUC:", roc_auc_score(y_test_vids, scores))

print("Sample normal keys:",   normal_keys[:3])
print("Sample abnormal keys:", abnormal_keys[:3])

# ── 5. FLATTEN SEGMENTS FOR TRAINING (no leakage — already split by video) ───

X_train_flat = X_train_norm.reshape(-1, 1024)  # (N_train*25, 1024)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_flat),
    torch.zeros(len(X_train_flat))
)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# ── 6. TRAIN VAE ──────────────────────────────────────────────────────────────

device = torch.device("cpu")

model = VariationalAutoencoder(
#    q_sigma=0.05, # try .0001, .001, .01
    n_dims_code=32,
    n_dims_data=1024,
    hidden_layer_sizes=[512]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class Args:
    n_mc_samples = 1

args = Args()

for epoch in range(1, 101):
    model.train_for_one_epoch_of_gradient_update_steps(
        optimizer, train_loader, device, epoch, args
    )

# ── 7. EVALUATE PER VIDEO — PRESERVING TEMPORAL STRUCTURE ────────────────────

model.eval()
video_recon_errors = []

for vid_feats in X_test_norm:
    # vid_feats shape: (25, 1024) — all segments for one video
    x = torch.FloatTensor(vid_feats).to(device)  # (25, 1024)

    with torch.no_grad():
        x_recon, mu, log_var = model(x)
    print("mu:  ", mu.mean().item(), mu.std().item())
    print("std: ", torch.exp(0.5*log_var).mean().item())    
    # Reconstruction error per segment: (25,)
    seg_errors = torch.mean((x - x_recon) ** 2, dim=1).numpy()

    # Aggregate across segments — try mean, max, or top-k
    video_recon_errors.append(seg_errors.mean())   # or .max()

video_recon_errors = np.array(video_recon_errors)

# ── 8. RESULTS ────────────────────────────────────────────────────────────────

print("\nReconstruction error by class:")
print(f"  Normal   mean: {video_recon_errors[y_test_vids==0].mean():.6f}")
print(f"  Abnormal mean: {video_recon_errors[y_test_vids==1].mean():.6f}")

auc = roc_auc_score(y_test_vids, video_recon_errors)
print(f"\nAUC-ROC: {auc:.4f}")

# Best threshold by F1
fpr, tpr, thresholds = roc_curve(y_test_vids, video_recon_errors)
best_thresh, best_f1 = 0, 0
for t in thresholds:
    preds = (video_recon_errors >= t).astype(int)
    f1 = f1_score(y_test_vids, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best threshold: {best_thresh:.6f}")
print(f"Best F1:        {best_f1:.4f}")
print(classification_report(
    y_test_vids,
    (video_recon_errors >= best_thresh).astype(int),
    target_names=['normal', 'abnormal']
))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test_vids, video_recon_errors)
auc = roc_auc_score(y_test_vids, video_recon_errors)


plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"VAE + TSN BN-Inception (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — VAE Anomaly Detection on TAD")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print(f"Normal test:   {len(normal_test_keys)}")   # expect ~50
print(f"Abnormal test: {len(abnormal_keys)}")       # this is all abnormal videos