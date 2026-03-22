import torch
import pretrainedmodels
import torch.nn as nn
import numpy as np
import os
import pickle
from PIL import Image
from torchvision import transforms

# ── LOAD BN-INCEPTION ─────────────────────────────────────────────────────────
model = pretrainedmodels.__dict__['bninception'](pretrained='imagenet')
model.eval()

# Use a forward hook on the global average pool layer to get 1024-d features
features_out = {}

def hook_fn(module, input, output):
    features_out['feat'] = output.view(output.size(0), -1)

# Register hook on the last pooling layer
model.global_pool.register_forward_hook(hook_fn)

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[104/255, 117/255, 128/255],
        std=[1/255, 1/255, 1/255]
    )
])

# ── EXTRACT FEATURES FOR ONE VIDEO ───────────────────────────────────────────
def extract_video_features(frame_dir, num_segments=25):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    total = len(frames)

    if total == 0:
        return None

    indices = np.linspace(0, total - 1, num_segments, dtype=int)
    sampled = [frames[i] for i in indices]

    imgs = []
    for fname in sampled:
        img = Image.open(os.path.join(frame_dir, fname)).convert('RGB')
        imgs.append(transform(img))

    batch = torch.stack(imgs)  # (25, 3, 224, 224)

    with torch.no_grad():
        model(batch)  # forward pass triggers the hook

    feat = features_out['feat']  # (25, 1024)
    return feat.numpy()

# ── EXTRACT ALL VIDEOS ────────────────────────────────────────────────────────
output_path = "/Users/pascaleleone/Desktop/Tufts CS/CS145/project/cs145_final_project/tad_rgb_features.pkl"
rawframes_root = "/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/frames"

features = {}
label_map = {"abnormal": 1, "normal": 0}

if not os.path.exists(output_path):
    features = {}
    label_map = {"abnormal": 1, "normal": 0}

    for class_name in label_map:
        class_dir = os.path.join(rawframes_root, class_name)
        videos = sorted(os.listdir(class_dir))
        print(f"Processing {len(videos)} {class_name} videos...")

        for i, video_name in enumerate(videos):
            video_dir = os.path.join(class_dir, video_name)
            if not os.path.isdir(video_dir):
                continue
            feat = extract_video_features(video_dir, num_segments=25)
            if feat is not None:
                features[f"{class_name}/{video_name}"] = feat
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(videos)} done")

    with open(output_path, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved {len(features)} videos to {output_path}")

else:
    print("Features already extracted, loading from cache...")
    with open(output_path, "rb") as f:
        features = pickle.load(f)

# ── INSPECT ───────────────────────────────────────────────────────────────────
for k, v in list(features.items())[:5]:
    print(k, v.shape)  # expected: (25, 1024)

import numpy as np
import pickle


with open(output_path, "rb") as f:
    features = pickle.load(f)

# ── BASIC INFO ────────────────────────────────────────────────────────────────
print(f"Total videos: {len(features)}")

# ── SHAPE OF FEATURES ─────────────────────────────────────────────────────────
sample_key = list(features.keys())[0]
sample_feat = features[sample_key]
print(f"\nSample key: {sample_key}")
print(f"Feature shape: {sample_feat.shape}")   # expected (25, 1024)
print(f"Feature dtype: {sample_feat.dtype}")

# ── CLASS BREAKDOWN ───────────────────────────────────────────────────────────
abnormal = {k: v for k, v in features.items() if k.startswith("abnormal")}
normal   = {k: v for k, v in features.items() if k.startswith("normal")}
print(f"\nAbnormal videos: {len(abnormal)}")
print(f"Normal videos:   {len(normal)}")

# ── VALUE STATS ───────────────────────────────────────────────────────────────
all_feats = np.stack(list(features.values()))  # (N, 25, 1024)
print(f"\nFull feature array shape: {all_feats.shape}")
print(f"Mean:  {all_feats.mean():.4f}")
print(f"Std:   {all_feats.std():.4f}")
print(f"Min:   {all_feats.min():.4f}")
print(f"Max:   {all_feats.max():.4f}")

# ── PER-VIDEO SUMMARY (first 5) ───────────────────────────────────────────────
print("\nFirst 5 videos:")
for k, v in list(features.items())[:5]:
    print(f"  {k}: shape={v.shape}, mean={v.mean():.3f}, std={v.std():.3f}")