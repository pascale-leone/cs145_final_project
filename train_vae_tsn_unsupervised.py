import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from starter_vae import VariationalAutoencoder
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── 1. LOAD FEATURES ─────────────────────────────────────────────────────────

with open("tad_rgb_features.pkl", "rb") as f:
    features = pickle.load(f)

video_keys = list(features.keys())
video_labels = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

# ── 2. SPLIT AT VIDEO LEVEL ───────────────────────────────────────────────────



train_keys, test_keys = train_test_split(
    video_keys, test_size=0.2, random_state=42
)


print(f"Train videos: {len(train_keys)}")
print(f"Test videos:       {len(test_keys)}")


# ── 3. BUILD ARRAYS ──────────────────────────────────────────────────────────

X_train_vids = np.stack([features[k] for k in train_keys])   # (N_train, 25, 1024)
X_test_vids  = np.stack([features[k] for k in test_keys])    # (N_test,  25, 1024)
y_test_vids  = np.array([1 if k.startswith("abnormal") else 0 for k in test_keys])

# ── 4. Z-SCORE NORMALIZE USING TRAIN STATS ───────────────────────────────────

X_train_flat_raw = X_train_vids.reshape(-1, 1024)
feat_mean = X_train_flat_raw.mean(axis=0)
feat_std  = X_train_flat_raw.std(axis=0) + 1e-8  # avoid division by zero

X_train_norm = (X_train_vids - feat_mean) / feat_std
X_test_norm  = (X_test_vids  - feat_mean) / feat_std

# ── 5. FLATTEN SEGMENTS FOR TRAINING ─────────────────────────────────────────

X_train_flat = X_train_norm.reshape(-1, 1024)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_flat),
    torch.zeros(len(X_train_flat))
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ── 6. TRAIN VAE ──────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VariationalAutoencoder(
    n_dims_code=32,
    n_dims_data=1024,
    hidden_layer_sizes=[512, 256]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100
for epoch in range(1, n_epochs + 1):
    model.train_for_one_epoch(optimizer, train_loader, device, epoch)

model.save_to_file("vae_baseline.pt")
print("Model saved to vae_baseline.pt")

# ── 7. EVALUATE ──────────────────────────────────────────────────────────────

model.eval()
video_recon_errors = []

for vid_feats in X_test_norm:
    x = torch.FloatTensor(vid_feats).to(device)  # (25, 1024)
    with torch.no_grad():
        x_recon, mu, log_var = model(x)
    seg_errors = torch.mean((x - x_recon) ** 2, dim=1).cpu().numpy()
    video_recon_errors.append(seg_errors.max())

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

# ── 9. ROC CURVE PLOT ────────────────────────────────────────────────────────

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"VAE baseline (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — VAE Anomaly Detection on TAD Unsupervised")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_unsupervised.png")
plt.show()
