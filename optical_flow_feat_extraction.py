import cv2
import numpy as np
import os
import pickle

def compute_flow_features_for_video(frame_dir, num_segments=25):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    total = len(frames)
    if total < 2:
        return None

    # Sample pairs of consecutive frames
    indices = np.linspace(0, total - 2, num_segments, dtype=int)
    
    flows = []
    for i in indices:
        img1 = cv2.imread(os.path.join(frame_dir, frames[i]))
        img2 = cv2.imread(os.path.join(frame_dir, frames[i+1]))
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )  # shape: (H, W, 2) — x and y flow
        
        # Summarize into a feature vector
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        feat = np.array([
            magnitude.mean(),
            magnitude.std(),
            magnitude.max(),
            np.percentile(magnitude, 90),
            angle.mean(),
            angle.std(),
        ])
        flows.append(feat)
    
    return np.stack(flows)  # (25, 6)


# ── EXTRACT ALL VIDEOS ────────────────────────────────────────────────────────
rawframes_root = "/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/frames"
output_path = "tad_flow_features.pkl"

features = {}
for class_name in ["normal", "abnormal"]:
    class_dir = os.path.join(rawframes_root, class_name)
    videos = sorted(os.listdir(class_dir))
    print(f"Processing {len(videos)} {class_name} videos...")
    
    for i, video_name in enumerate(videos):
        video_dir = os.path.join(class_dir, video_name)
        if not os.path.isdir(video_dir):
            continue
        feat = compute_flow_features_for_video(video_dir)
        if feat is not None:
            features[f"{class_name}/{video_name}"] = feat
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(videos)} done")

with open(output_path, "wb") as f:
    pickle.dump(features, f)
print(f"Saved {len(features)} videos")