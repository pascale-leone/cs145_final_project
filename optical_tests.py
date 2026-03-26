import pickle
import numpy as np
# Check if flow features separate the classes at all
with open("tad_flow_features.pkl", "rb") as f:
    flow_features = pickle.load(f)

video_keys  = list(flow_features.keys())
y           = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])
X           = np.stack([flow_features[k].mean(axis=0) for k in video_keys])  # (N, 6)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
norms    = np.linalg.norm(X_scaled, axis=1)
print("Flow feature AUC:", roc_auc_score(y, norms))

# Inspect flow features to make sure they're not garbage
normal_keys   = [k for k in video_keys if k.startswith("normal")]
abnormal_keys = [k for k in video_keys if k.startswith("abnormal")]

normal_feats   = np.stack([flow_features[k].mean(axis=0) for k in normal_keys])
abnormal_feats = np.stack([flow_features[k].mean(axis=0) for k in abnormal_keys])

feature_names = ['mag_mean', 'mag_std', 'mag_max', 'mag_p90', 'angle_mean', 'angle_std']

print("Feature-by-feature breakdown:")
print(f"{'feature':<12} {'normal_mean':>12} {'abnormal_mean':>14} {'diff':>8}")
for i, name in enumerate(feature_names):
    n = normal_feats[:, i].mean()
    a = abnormal_feats[:, i].mean()
    print(f"{name:<12} {n:>12.4f} {a:>14.4f} {a-n:>8.4f}")

# Also check a few individual videos
print("\nSample normal flow magnitudes (mean per segment):")
for k in normal_keys[:3]:
    print(f"  {k}: {flow_features[k][:, 0]}")  # mag_mean per segment

print("\nSample abnormal flow magnitudes (mean per segment):")
for k in abnormal_keys[:3]:
    print(f"  {k}: {flow_features[k][:, 0]}")  # mag_mean per segment


# Instead of mean(axis=0), use max and high percentiles to catch spikes
def video_to_score(feat_25x6):
    mag_per_segment = feat_25x6[:, 0]  # mag_mean per segment
    return np.array([
        mag_per_segment.max(),
        np.percentile(mag_per_segment, 90),
        mag_per_segment.std(),
        feat_25x6[:, 2].max(),   # mag_max (peak pixel flow)
    ])

X = np.stack([video_to_score(flow_features[k]) for k in video_keys])
y = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Try each feature individually
feature_names = ['mag_mean_max', 'mag_mean_p90', 'mag_mean_std', 'mag_max_peak']
for i, name in enumerate(feature_names):
    auc = roc_auc_score(y, X_scaled[:, i])
    print(f"{name}: AUC = {auc:.4f}  (flipped: {1-auc:.4f})")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def video_to_features(feat_25x6):
    mag_mean_per_seg = feat_25x6[:, 0]   # mag_mean per segment
    mag_max_per_seg  = feat_25x6[:, 2]   # mag_max per segment
    mag_std_per_seg  = feat_25x6[:, 1]   # mag_std per segment

    return np.array([
        mag_max_per_seg.max(),            # peak pixel flow — your 0.82 signal
        mag_max_per_seg.mean(),           
        np.percentile(mag_max_per_seg, 90),
        mag_max_per_seg.std(),            # how much flow spikes vary
        mag_mean_per_seg.std(),           
        # Spike detection: how many segments exceed a high threshold
        (mag_max_per_seg > mag_max_per_seg.mean() + mag_max_per_seg.std()).sum(),
    ])

X = np.stack([video_to_features(flow_features[k]) for k in video_keys])
y = np.array([1 if k.startswith("abnormal") else 0 for k in video_keys])

X_scaled = StandardScaler().fit_transform(X)

# Individual AUCs
names = ['mag_max_peak', 'mag_max_mean', 'mag_max_p90', 'mag_max_std', 'mag_mean_std', 'spike_count']
for i, name in enumerate(names):
    auc = roc_auc_score(y, X_scaled[:, i])
    print(f"{name}: AUC = {auc:.4f}")

# Combined score
combined = X_scaled.mean(axis=1)
print(f"\nCombined mean: AUC = {roc_auc_score(y, combined):.4f}")

# Weighted toward best signal
weighted = X_scaled[:, 0] * 2 + X_scaled[:, 1] + X_scaled[:, 3]
print(f"Weighted combo: AUC = {roc_auc_score(y, weighted):.4f}")

from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Try all combinations of the top 4 features
top_indices = [0, 1, 2, 3]  # mag_max_peak, mag_max_mean, mag_max_p90, mag_max_std
top_names   = ['mag_max_peak', 'mag_max_mean', 'mag_max_p90', 'mag_max_std']

print("All combinations:")
for r in range(1, 5):
    for combo in combinations(range(4), r):
        X_combo = X_scaled[:, combo]
        if X_combo.ndim == 1:
            X_combo = X_combo.reshape(-1, 1)
        # Use logistic regression with cross-validation
        scores = cross_val_score(
            LogisticRegression(), X_combo, y,
            cv=5, scoring='roc_auc'
        )
        names_combo = [top_names[i] for i in combo]
        print(f"  {str(names_combo):<60} AUC = {scores.mean():.4f} ± {scores.std():.4f}")

