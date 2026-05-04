import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

vae_scores  = np.load("results/scores/vae_test_scores.npy")
ldm_scores  = np.load("results/scores/ldm_test_scores.npy")
y_test      = np.load("results/scores/test_labels.npy")

def plot_reliability(scores, labels, model_name, ax, n_bins=10):
    s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    frac_pos, mean_pred = calibration_curve(labels, s,
                                            n_bins=n_bins,
                                            strategy='uniform')
    ece = np.mean(np.abs(frac_pos - mean_pred))
    ax.plot(mean_pred, frac_pos, 'o-', label=f'{model_name} (ECE={ece:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Perfect')
    ax.set_xlabel('Mean predicted score')
    ax.set_ylabel('Fraction anomalies')
    ax.set_title(f'Reliability diagram — {model_name}')
    ax.legend()
    return ece

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ece_vae = plot_reliability(vae_scores, y_test, 'VAE',  axes[0])
ece_ldm = plot_reliability(ldm_scores, y_test, 'LDM', axes[1])
plt.tight_layout()
plt.savefig("results/figures/reliability_diagrams.png", dpi=150)
print(f"VAE ECE: {ece_vae:.4f}")
print(f"LDM ECE: {ece_ldm:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_score_cdfs(scores, labels, model_name):
    normal_scores   = np.sort(scores[labels == 0])
    abnormal_scores = np.sort(scores[labels == 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # CDF
    axes[0].plot(normal_scores,   np.linspace(0, 1, len(normal_scores)),
                 label='Normal')
    axes[0].plot(abnormal_scores, np.linspace(0, 1, len(abnormal_scores)),
                 label='Abnormal')
    axes[0].set_xlabel('Score'); axes[0].set_ylabel('Cumulative fraction')
    axes[0].set_title(f'{model_name} — score CDFs')
    axes[0].legend()

    # KDE
    
    xs = np.linspace(scores.min(), scores.max(), 300)
    axes[1].plot(xs, gaussian_kde(normal_scores)(xs),   label='Normal')
    axes[1].plot(xs, gaussian_kde(abnormal_scores)(xs), label='Abnormal')
    axes[1].set_xlabel('Score'); axes[1].set_ylabel('Density')
    axes[1].set_title(f'{model_name} — score distributions')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'results/figures/score_cdfs_{model_name.lower()}.png', dpi=150)

plot_score_cdfs(vae_scores, y_test, 'VAE')
plot_score_cdfs(ldm_scores, y_test, 'LDM')