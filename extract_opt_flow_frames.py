import cv2
import os
import numpy as np
from PIL import Image

rawframes_root = "/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/frames"
flow_root      = "/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/flow"

def compute_tvl1_flow(frame_dir, flow_out_dir):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if len(frames) < 2:
        return

    os.makedirs(flow_out_dir, exist_ok=True)

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    for i in range(len(frames) - 1):
        img1 = cv2.imread(os.path.join(frame_dir, frames[i]))
        img2 = cv2.imread(os.path.join(frame_dir, frames[i + 1]))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        flow = tvl1.calc(gray1, gray2, None)  # (H, W, 2)

        # Clip and normalise to [0, 255] for saving as uint8 images
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        def normalise(f):
            f = np.clip(f, -20, 20)
            f = (f + 20) * (255 / 40)
            return f.astype(np.uint8)

        x_img = Image.fromarray(normalise(flow_x))
        y_img = Image.fromarray(normalise(flow_y))

        idx = str(i + 1).zfill(5)
        x_img.save(os.path.join(flow_out_dir, f'flow_x_{idx}.jpg'))
        y_img.save(os.path.join(flow_out_dir, f'flow_y_{idx}.jpg'))


# ── RUN OVER ALL VIDEOS ───────────────────────────────────────────────────────
for class_name in ["abnormal", "normal"]:
    class_dir      = os.path.join(rawframes_root, class_name)
    class_flow_dir = os.path.join(flow_root, class_name)
    videos = sorted(os.listdir(class_dir))
    print(f"Processing {len(videos)} {class_name} videos...")

    for i, video_name in enumerate(videos):
        video_dir      = os.path.join(class_dir, video_name)
        video_flow_dir = os.path.join(class_flow_dir, video_name)
        if not os.path.isdir(video_dir):
            continue
        compute_tvl1_flow(video_dir, video_flow_dir)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(videos)} done")

print("Flow extraction complete.")