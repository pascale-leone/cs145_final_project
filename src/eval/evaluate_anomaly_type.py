import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path

vae_scores  = np.load("vae_test_scores_2.npy")
ldm_scores  = np.load("ldm_test_scores_2.npy")
y_test      = np.load("test_labels_2.npy")
test_keys = np.load("test_keys.npy")

def parse_category(key):
    # key looks like 'abnormal/01_Accident_001' or 'normal/Normal_001'
    stem = Path(key).stem  # strips any extension if present
    name = Path(key).name  # e.g. '01_Accident_001'
    if name.startswith('Normal'):
        return 'Normal'
    parts = name.split('_')
    return '_'.join(parts[:-1])  # e.g. '01_Accident', '02_IllegalTurn'

rgb_df = pd.DataFrame({
    'key': test_keys,
    'vae_score': vae_scores,
    'ldm_score': ldm_scores,
    'label': y_test,
    'category': [parse_category(k) for k in test_keys]
})

# Per-category stats
baseline = y_test.mean()
results = []
for cat, group in rgb_df[rgb_df.label == 1].groupby('category'):
    normal = rgb_df[rgb_df.label == 0]
    sub = pd.concat([group, normal])
    auprc_vae = average_precision_score(sub.label, sub.vae_score)
    auprc_ldm = average_precision_score(sub.label, sub.ldm_score)
    auroc_vae = roc_auc_score(sub.label, sub.vae_score)
    auroc_ldm = roc_auc_score(sub.label, sub.ldm_score)
    local_baseline = group.shape[0] / len(sub)  # category-specific baseline
    normalized_gain_vae = (auprc_vae - local_baseline) / (1 - local_baseline)
    normalized_gain_ldm = (auprc_ldm - local_baseline) / (1 - local_baseline)
    results.append({
        'category': cat,
        'n_samples': len(group),
        #'mean_vae_score': group.vae_score.mean(),
        #'mean_ldm_score': group.ldm_score.mean(),
        #'auprc_vae': auprc_vae,
        #'auprc_ldm': auprc_ldm,
        'auroc_vae': auroc_vae,
        'auroc_ldm': auroc_ldm,
    #    'local_baseline': local_baseline,
    #    'normalized_gain_vae': normalized_gain_vae,
    #    'normalized_gain_ldm': normalized_gain_ldm,
    })

results_df = pd.DataFrame(results).sort_values('category')
print(results_df.to_latex(index=False))

# normal_scores = ldm_scores_twostream[y_test == 0]
# abnormal_scores = ldm_scores_twostream[y_test == 1]
# print(f"Normal mean:   {normal_scores.mean():.2f}")
# print(f"Abnormal mean: {abnormal_scores.mean():.2f}")

