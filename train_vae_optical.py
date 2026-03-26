import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from update_vae import VariationalAutoencoder
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── 1. LOAD FLOW FEATURES ─────────────────────────────────────────────────────
with open("tad_flow_features.pkl", "rb") as f:
    features = pickle.load(f)

video_keys    = list(features.keys())
video_labels  = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

# ── 2. EXTRACT mag_max TIME SERIES (25,) per video ───────────────────────────
# features[k] shape: (25, 6) — column 2 is mag_max per segment
X_all = np.stack([features[k][:, 2] for k in video_keys])  # (N, 25)

# ── 3. SPLIT AT VIDEO LEVEL ───────────────────────────────────────────────────
normal_keys   = [k for k, l in zip(video_keys, video_labels) if l == 0]
abnormal_keys = [k for k, l in zip(video_keys, video_labels) if l == 1]

train_keys, normal_test_keys = train_test_split(
    normal_keys, test_size=0.2, random_state=42
)
test_keys = normal_test_keys + abnormal_keys
y_test    = np.array([0 if k in normal_test_keys else 1 for k in test_keys])

print(f"Train videos (normal only): {len(train_keys)}")
print(f"Test  videos (normal):      {len(normal_test_keys)}")
print(f"Test  videos (abnormal):    {len(abnormal_keys)}")

# ── 4. BUILD ARRAYS ───────────────────────────────────────────────────────────
key_to_idx = {k: i for i, k in enumerate(video_keys)}

X_train = X_all[[key_to_idx[k] for k in train_keys]]       # (N_train, 25)
X_test  = X_all[[key_to_idx[k] for k in test_keys]]        # (N_test,  25)

# ── 5. NORMALIZE USING TRAIN STATS ONLY ──────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)                     # (N_train, 25)
X_test  = scaler.transform(X_test)                          # (N_test,  25)

# ── 6. DATALOADER ─────────────────────────────────────────────────────────────
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.zeros(len(X_train))
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ── 7. TRAIN VAE ──────────────────────────────────────────────────────────────
device = torch.device("cpu")

model = VariationalAutoencoder(
    n_dims_code=4,
    n_dims_data=25,
    hidden_layer_sizes=[64, 32]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class Args:
    n_mc_samples = 1

args = Args()

for epoch in range(1, 201):
    model.train_for_one_epoch_of_gradient_update_steps(
        optimizer, train_loader, device, epoch, args
    )

# ── 8. EVALUATE ───────────────────────────────────────────────────────────────
model.eval()
X_test_tensor = torch.FloatTensor(X_test)

with torch.no_grad():
    x_recon, mu, log_var = model(X_test_tensor)

recon_errors = torch.mean((X_test_tensor - x_recon) ** 2, dim=1).numpy()

print("\nReconstruction error by class:")
print(f"  Normal   mean: {recon_errors[y_test==0].mean():.6f}")
print(f"  Abnormal mean: {recon_errors[y_test==1].mean():.6f}")

auc = roc_auc_score(y_test, recon_errors)
print(f"\nAUC-ROC: {auc:.4f}")

# ── 9. BEST THRESHOLD BY F1 ───────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, recon_errors)
best_thresh, best_f1 = 0, 0
for t in thresholds:
    preds = (recon_errors >= t).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1   = f1
        best_thresh = t

print(f"Best threshold: {best_thresh:.6f}")
print(f"Best F1:        {best_f1:.4f}")
print(classification_report(
    y_test,
    (recon_errors >= best_thresh).astype(int),
    target_names=['normal', 'abnormal'],
    zero_division=0
))

# ── 10. ROC CURVE ─────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"VAE on Flow Features (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — VAE Anomaly Detection on TAD")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_flow.png")
plt.show()