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

# ── SETUP: clone tsn-pytorch and add to path ──────────────────────────────────
# Run once in terminal:
#   git clone --recursive https://github.com/yjxiong/tsn-pytorch
# Then set this path:
TSN_ROOT = os.path.expanduser("~/libs/tsn-pytorch")
sys.path.insert(0, TSN_ROOT)
os.chdir(TSN_ROOT)

from models import TSN
from transforms import (Stack, ToTorchFormatTensor, GroupNormalize,
                        GroupScale, GroupCenterCrop)

# ── BUILD TSN MODELS ──────────────────────────────────────────────────────────
NUM_SEGMENTS = 25
FLOW_LENGTH  = 5   # TSN default: stack 5 consecutive x/y pairs → 10 channels

rgb_tsn = TSN(num_class=101, num_segments=NUM_SEGMENTS, modality='RGB',
              base_model='BNInception', consensus_type='avg',
              dropout=0.0)  # dropout=0 so base_model.fc stays linear

flow_tsn = TSN(num_class=101, num_segments=NUM_SEGMENTS, modality='Flow',
               base_model='BNInception', new_length=FLOW_LENGTH,
               consensus_type='avg', dropout=0.0)

rgb_tsn.eval()
flow_tsn.eval()

# ── HOOKS: tap pool layer for 1024-d features ─────────────────────────────────
features_out = {}

def make_hook(key):
    def hook_fn(module, input, output):
        features_out[key] = output.view(output.size(0), -1)
    return hook_fn

rgb_tsn.base_model.global_pool.register_forward_hook(make_hook('rgb'))
flow_tsn.base_model.global_pool.register_forward_hook(make_hook('flow'))

# ── TRANSFORMS (matching TSN's input_mean / input_std from models.py) ─────────
# RGB: mean=[104, 117, 128], std=[1]  (BNInception convention)
rgb_transform = transforms.Compose([
    GroupScale(256),
    GroupCenterCrop(224),
    Stack(roll=True),                          # list of PIL → (H, W, 3*N)
    ToTorchFormatTensor(div=False),            # → (3*N, H, W), keeps [0,255]
    GroupNormalize([104, 117, 128], [1, 1, 1]) # TSN's exact RGB norm
])

# Flow: mean=[128], std=[1] per channel, 10 channels total
flow_transform = transforms.Compose([
    GroupScale(256),
    GroupCenterCrop(224),
    Stack(roll=False),
    ToTorchFormatTensor(div=False),
    GroupNormalize([128] * (2 * FLOW_LENGTH), [1] * (2 * FLOW_LENGTH))
])

# ── LOAD RGB SNIPPET (single frame → list of 1 PIL image) ────────────────────
def load_rgb_frame(frame_path):
    return [Image.open(frame_path).convert('RGB')]

# ── LOAD FLOW SNIPPET (L consecutive x/y pairs → list of 2L grayscale images) ─
def load_flow_snippet(flow_dir, start_idx, L=5):
    """Returns list of 2L PIL grayscale images: [x1, y1, x2, y2, ..., xL, yL]"""
    imgs = []
    for i in range(L):
        idx = str(start_idx + i).zfill(5)
        x_path = os.path.join(flow_dir, f'flow_x_{idx}.jpg')
        y_path = os.path.join(flow_dir, f'flow_y_{idx}.jpg')
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            return None
        imgs.append(Image.open(x_path).convert('L'))
        imgs.append(Image.open(y_path).convert('L'))
    return imgs  # length = 2*L

# ── EXTRACT FEATURES FOR ONE VIDEO ───────────────────────────────────────────
def extract_video_features(frame_dir, flow_dir, num_segments=NUM_SEGMENTS):
    # ── RGB ───────────────────────────────────────────────────────────────────
    rgb_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    total_rgb  = len(rgb_frames)
    if total_rgb == 0:
        return None

    rgb_indices = np.linspace(0, total_rgb - 1, num_segments, dtype=int)

    # TSN transform expects a list of PIL images (one per segment for RGB)
    rgb_pil_list = [Image.open(os.path.join(frame_dir, rgb_frames[i])).convert('RGB')
                    for i in rgb_indices]
    rgb_tensor = rgb_transform(rgb_pil_list)           # (3*num_segments, 224, 224)
    rgb_batch  = rgb_tensor.view(num_segments, 3, 224, 224)  # (25, 3, 224, 224)

    # ── FLOW ──────────────────────────────────────────────────────────────────
    flow_x_files = sorted([f for f in os.listdir(flow_dir) if 'flow_x' in f])
    total_flow   = len(flow_x_files)
    if total_flow < FLOW_LENGTH:
        return None

    # Sample segment start indices leaving room for L consecutive frames
    flow_indices = np.linspace(0, total_flow - FLOW_LENGTH, num_segments, dtype=int)

    flow_pil_list = []  # will be 25 * 2*L = 250 images
    for start in flow_indices:
        snippet = load_flow_snippet(flow_dir, start + 1, L=FLOW_LENGTH)  # 1-indexed
        if snippet is None:
            return None
        flow_pil_list.extend(snippet)

    flow_tensor = flow_transform(flow_pil_list)  # (2*L*num_segments, 224, 224)
    flow_batch  = flow_tensor.view(num_segments, 2 * FLOW_LENGTH, 224, 224)  # (25, 10, 224, 224)

    # ── FORWARD ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        # TSN forward expects (batch*segments, C, H, W) — our batch=1 so just pass directly
        rgb_tsn.base_model(rgb_batch)
        flow_tsn.base_model(flow_batch)

    rgb_feat  = features_out['rgb'].numpy()   # (25, 1024)
    flow_feat = features_out['flow'].numpy()  # (25, 1024)

    return np.concatenate([rgb_feat, flow_feat], axis=1)  # (25, 2048)

# ── PATHS ─────────────────────────────────────────────────────────────────────
output_path    = "/Users/pascaleleone/Desktop/Tufts CS/CS145/project/cs145_final_project/tad_twostream_features_v3.pkl"
rawframes_root = "/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/frames"
flow_root      = "/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/flow"

label_map = {"abnormal": 1, "normal": 0}

if not os.path.exists(output_path):
    features = {}

    for class_name in label_map:
        class_dir      = os.path.join(rawframes_root, class_name)
        class_flow_dir = os.path.join(flow_root, class_name)
        videos = sorted(os.listdir(class_dir))
        print(f"Processing {len(videos)} {class_name} videos...")

        for i, video_name in enumerate(videos):
            video_dir      = os.path.join(class_dir, video_name)
            video_flow_dir = os.path.join(class_flow_dir, video_name)
            if not os.path.isdir(video_dir) or not os.path.isdir(video_flow_dir):
                continue
            feat = extract_video_features(video_dir, video_flow_dir)
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


for k, v in list(features.items())[:5]:
    print(k, v.shape)  # expected: (25, 1024)



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