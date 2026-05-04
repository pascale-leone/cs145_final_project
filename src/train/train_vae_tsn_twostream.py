import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.starter_vae import VariationalAutoencoder
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── 1. LOAD FEATURES ─────────────────────────────────────────────────────────

with open("data/features/tad_twostream_features_32.pkl", "rb") as f:
    features = pickle.load(f)

video_keys = list(features.keys())
video_labels = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

# ── 2. SPLIT AT VIDEO LEVEL (Train, Test, Val) ───────────────────────────────────────────────────

normal_keys   = [k for k, l in zip(video_keys, video_labels) if l == 0]
abnormal_keys = [k for k, l in zip(video_keys, video_labels) if l == 1]

# 75% of normal data for train
train_keys, normal_test_val_keys = train_test_split(
    normal_keys, test_size=0.25, random_state=42
)

normal_val_keys, normal_test_keys = train_test_split(
    normal_test_val_keys, test_size=0.5, random_state=42
)

abnormal_val_keys, abnormal_test_keys = train_test_split(
    abnormal_keys, test_size=0.5, random_state=42
)

val_keys = normal_val_keys + abnormal_val_keys
test_keys = normal_test_keys + abnormal_test_keys

print(f"Train videos (normal only): {len(train_keys)}")
print(f"Test videos (normal):       {len(normal_test_keys)}")
print(f"Test videos (abnormal):     {len(abnormal_test_keys)}")
print(f"Val videos (normal):       {len(normal_val_keys)}")
print(f"Val videos (abnormal):     {len(abnormal_val_keys)}")

# ── 3. BUILD ARRAYS ──────────────────────────────────────────────────────────

X_train_vids = np.stack([features[k] for k in train_keys])   # (N_train, 25, 2048)
X_test_vids  = np.stack([features[k] for k in test_keys])    # (N_test,  25, 2048)
y_test_vids  = np.array([0 if k in normal_test_keys else 1 for k in test_keys])
X_val_vids = np.stack([features[k] for k in val_keys])
y_val_vids  = np.array([0 if k in normal_val_keys else 1 for k in val_keys])

# ── 4. Z-SCORE NORMALIZE USING TRAIN STATS ───────────────────────────────────

X_train_flat_raw = X_train_vids.reshape(-1, 2048)
feat_mean = X_train_flat_raw.mean(axis=0)
feat_std  = X_train_flat_raw.std(axis=0) + 1e-8  # avoid division by zero

X_train_norm = (X_train_vids - feat_mean) / feat_std
X_test_norm  = (X_test_vids  - feat_mean) / feat_std
X_val_norm  = (X_val_vids  - feat_mean) / feat_std

# ── 5. FLATTEN SEGMENTS FOR TRAINING ─────────────────────────────────────────

X_train_flat = X_train_norm.reshape(-1, 2048)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_flat),
    torch.zeros(len(X_train_flat))
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ── 6. TRAIN VAE ──────────────────────────────────────────────────────────────
sigma2_values = np.logspace(-3,1,10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_epochs=100
models, all_auc, all_beta = [], [], []
for sigma2 in sigma2_values:
    # Train
    torch.manual_seed(42)
    np.random.seed(42)
    model = VariationalAutoencoder(
        n_dims_code=32,
        n_dims_data=2048,
        hidden_layer_sizes=[512, 256]
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, n_epochs + 1):
        model.train_for_one_epoch(optimizer, train_loader, device, epoch, sigma2=sigma2)
    
    models.append(model)
    # Precompute val scores
    model.eval()
    val_mean, val_max, val_k = [], [], []
    val_elbo = []
    for vid_feats in X_val_norm:
        x = torch.FloatTensor(vid_feats).to(device)
        D = x.shape[1]
        with torch.no_grad():
            x_recon, mu, log_var = model(x)

        recon_log_likelihood = -torch.sum((x - x_recon) ** 2, dim=1) / (2 * sigma2) \
                       - (D / 2) * np.log(2 * np.pi * sigma2)  # shape (32,)

        kl_per_seg = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        ELBO = recon_log_likelihood - kl_per_seg  # higher = more normal

        mean_score = (-ELBO).mean().item()
        max_score  = (-ELBO).max().item()
        topk_score = np.mean(np.sort((-ELBO).cpu().numpy())[-5:])
        val_mean.append(mean_score)
        val_max.append(max_score)
        val_k.append(topk_score)

        k = 5
        video_score = np.mean(np.sort(-ELBO.cpu().numpy())[-k:])
        val_elbo.append(video_score)

    auc_mean = roc_auc_score(y_val_vids, val_mean)
    auc_max = roc_auc_score(y_val_vids, val_max)
    auc_k = roc_auc_score(y_val_vids, val_k)
    
    print(f'sigma2: {sigma2}, auc_mean: {auc_mean:.4f}, auc_max: {auc_max:.4f}, auc_k: {auc_k:.4f}')

    
    all_auc.append(auc_k)

best_model = models[np.argmax(all_auc)]
best_sigma2 = sigma2_values[np.argmax(all_auc)]

best_model.eval()
video_recon_errors = []


for vid_feats in X_test_norm:
    x = torch.FloatTensor(vid_feats).to(device)  # (32, 1024)
    D = x.shape[1]
    with torch.no_grad():
        x_recon, mu, log_var = best_model(x)

    recon_log_likelihood = -torch.sum((x - x_recon) ** 2, dim=1) / (2 * best_sigma2) \
                       - (D / 2) * np.log(2 * np.pi * best_sigma2)  # shape (25,)

    kl_per_seg = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

    ELBO = recon_log_likelihood - kl_per_seg  # higher = more normal

    topk = np.mean(np.sort((-ELBO).cpu().numpy())[-5:])
    video_recon_errors.append(topk)


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
plt.title("ROC Curve — VAE Anomaly Detection on TAD (two stream feature extraction)")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/roc_curve_twostream.png")
plt.show()
