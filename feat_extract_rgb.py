import sys
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from PIL import Image
import torchvision.transforms as transforms
import yaml

# patch before tsn imports
_yaml_load = yaml.load
yaml.load = lambda f, **kwargs: _yaml_load(f, Loader=yaml.SafeLoader)

TSN_ROOT = os.path.expanduser("~/libs/tsn-pytorch")
sys.path.insert(0, TSN_ROOT)
os.chdir(TSN_ROOT)

from models import TSN
from transforms import (Stack, ToTorchFormatTensor, GroupNormalize,
                        GroupScale, GroupCenterCrop)

# ── BUILD RGB TSN MODEL ONLY ──────────────────────────────────────────────────
NUM_SEGMENTS = 25

rgb_tsn = TSN(num_class=101, num_segments=NUM_SEGMENTS, modality='RGB',
              base_model='BNInception', consensus_type='avg', dropout=0.0)
rgb_tsn.eval()

# ── HOOK: tap pool layer for 1024-d features ──────────────────────────────────
features_out = {}

def make_hook(key):
    def hook_fn(module, input, output):
        features_out[key] = output.view(output.size(0), -1)
    return hook_fn

rgb_tsn.base_model.global_pool.register_forward_hook(make_hook('rgb'))

# ── RGB TRANSFORM ─────────────────────────────────────────────────────────────
rgb_transform = transforms.Compose([
    GroupScale(256),
    GroupCenterCrop(224),
    Stack(roll=True),
    ToTorchFormatTensor(div=False),
    GroupNormalize([104, 117, 128], [1, 1, 1])
])

# ── EXTRACT FEATURES FOR ONE VIDEO (RGB only) ─────────────────────────────────
def extract_video_features(frame_dir, num_segments=NUM_SEGMENTS):
    rgb_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    total_rgb  = len(rgb_frames)
    if total_rgb == 0:
        return None

    rgb_indices  = np.linspace(0, total_rgb - 1, num_segments, dtype=int)
    rgb_pil_list = [Image.open(os.path.join(frame_dir, rgb_frames[i])).convert('RGB')
                    for i in rgb_indices]

    rgb_tensor = rgb_transform(rgb_pil_list)               # (3*num_segments, 224, 224)
    rgb_batch  = rgb_tensor.view(num_segments, 3, 224, 224) # (25, 3, 224, 224)

    with torch.no_grad():
        rgb_tsn.base_model(rgb_batch)

    return features_out['rgb'].numpy()  # (25, 1024)

# ── PATHS ─────────────────────────────────────────────────────────────────────
output_path    = "/Users/pascaleleone/Desktop/Tufts CS/CS145/project/cs145_final_project/tad_rgb_features_v2.pkl"
rawframes_root = "/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/frames"

label_map = {"abnormal": 1, "normal": 0}

if not os.path.exists(output_path):
    features = {}

    for class_name in label_map:
        class_dir = os.path.join(rawframes_root, class_name)
        videos    = sorted(os.listdir(class_dir))
        print(f"Processing {len(videos)} {class_name} videos...")

        for i, video_name in enumerate(videos):
            video_dir = os.path.join(class_dir, video_name)
            if not os.path.isdir(video_dir):
                continue
            feat = extract_video_features(video_dir)
            if feat is not None:
                features[f"{class_name}/{video_name}"] = feat
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(videos)} done")

    with open(output_path, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved {len(features)} videos to {output_path}")

else:
    print("Loading cached features...")
    with open(output_path, "rb") as f:
        features = pickle.load(f)

# ── INSPECT ───────────────────────────────────────────────────────────────────
print(f"Total videos: {len(features)}")
sample_key  = list(features.keys())[0]
sample_feat = features[sample_key]
print(f"Sample: {sample_key} → shape={sample_feat.shape}, mean={sample_feat.mean():.3f}")