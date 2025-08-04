import torch
import numpy as np
import cv2
from torchvision.transforms import functional as F
import sys
sys.path.append('./RAFT/core')
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from glob import glob
import matplotlib.pyplot as plt
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--video_dir', type=str, required=True)
parser.add_argument('--save_plot_name', type=str, required=True)
args = parser.parse_args()

i = np.random.randint(1,4)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{i}"
os.environ["CUDA_LAUNCH_BLOCKING"] = f"{i}"

class CFG:
    save_syntax = args.model
    check_video = args.video_dir
    plot_name = args.save_plot_name

# Generate the JET colormap
jet_colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
# Create a lookup array instead of dictionary
color_to_class_array = np.full((256, 256, 256), -1, dtype=np.uint8)
for class_idx in range(256 // 25):
    scaled_value = class_idx * 25
    if scaled_value > 255:
        break
    color = tuple(jet_colormap[scaled_value][0])
    color_to_class_array[color[0], color[1], color[2]] = class_idx

def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')

# Initialize RAFT model
def initialize_raft_model():
    args = type('', (), {})()
    args.small = False
    args.mixed_precision = True
    args.alternate_corr = False
    args.dropout=0

    model = RAFT(args)
    model.load_state_dict(fix_key(torch.load('./RAFT/models/raft-things.pth')))
    model = model.eval().cuda()
    return model

# Compute optical flow using RAFT
def compute_optical_flow(model, frame1, frame2):
    frame1 = load_image(frame1)
    frame2 = load_image(frame2)

    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    
    with torch.no_grad():
        flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()

# Warp segmentation using optical flow
def warp_segmentation(segmentation, flow):
    h, w = segmentation.shape
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.stack(flow_map, axis=-1) + flow
    
    # Swap x and y coordinates for cv2.remap
    flow_map_y = flow_map[..., 1]  # y coordinates 
    flow_map_x = flow_map[..., 0]  # x coordinates
    
    # Clip to valid image coordinates
    flow_map_x = np.clip(flow_map_x, 0, w - 1)
    flow_map_y = np.clip(flow_map_y, 0, h - 1)
    
    # Convert to float32 for cv2.remap
    flow_map_x = flow_map_x.astype(np.float32)
    flow_map_y = flow_map_y.astype(np.float32)
    
    return cv2.remap(segmentation, flow_map_x, flow_map_y, interpolation=cv2.INTER_NEAREST)

# Calculate mean IoU between two segmentations
def calculate_miou(seg1, seg2):
    # クラスごとのIoUを計算
    unique_classes = [0,1,2,3,4,5,6,7,8,9]
    
    ious = []
    for cls in unique_classes:
        if cls == 0:
            intersection = np.logical_and(seg1 != 0, seg2 != 0).sum()
        else:
            intersection = np.logical_and(seg1 == cls, seg2 == cls).sum()
        union = np.logical_or(seg1 == cls, seg2 == cls).sum()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(np.nan)
    
    return np.array(ious) if ious else None


# Temporal consistency evaluation pipeline
def evaluate_temporal_consistency(model, video_frames, segmentations, save_plot_name=None):
    temporal_consistencies = []

    for i in tqdm(range(len(video_frames) - 1)):
    #for i in range(len(video_frames) - 1):
        frame1, frame2 = video_frames[i], video_frames[i + 1]
        seg1, seg2 = segmentations[i], segmentations[i + 1]
        seg1 = cv2.imread(seg1, cv2.IMREAD_GRAYSCALE)
        seg2 = cv2.imread(seg2, cv2.IMREAD_GRAYSCALE)
        seg1 = cv2.resize(seg1, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        seg2 = cv2.resize(seg2, (1920, 1080), interpolation=cv2.INTER_NEAREST)

        flow = compute_optical_flow(model, frame1, frame2)
        warped_seg2 = warp_segmentation(seg1, flow)

        temporal_consistencies.append(calculate_miou(warped_seg2, seg2))
    temporal_consistencies = np.array(temporal_consistencies)
    if save_plot_name is not None:
        mv_avg = []
        width = 10
        for i in range(temporal_consistencies.shape[0] - width):
            mv_avg.append(np.nanmean(temporal_consistencies[i:i+width,:], axis=0))
        plt.figure(figsize=(10, 5))
        plt.plot(mv_avg)
        plt.legend([f"{i}" for i in range(temporal_consistencies.shape[1])])
        os.makedirs(f"plot/{CFG.save_syntax}_1/{CFG.plot_name}", exist_ok=True)
        plt.savefig(f"plot/{CFG.save_syntax}_1/{CFG.plot_name}/{save_plot_name}.png")
        plt.close()
    mv_avg = np.array(mv_avg)
    np.save(f"plot/{CFG.save_syntax}_1/{CFG.plot_name}/{save_plot_name}.npy", mv_avg)
    return np.nanmean(temporal_consistencies, axis=0)

# Main function to run the evaluation
def main():
    dir_length = len(glob(f'{CFG.check_video}/rgb/*.png'))
    # Load video frames and segmentations (example paths)
    video_frames = [f'{CFG.check_video}/rgb/{i:09}.png' for i in range(0, dir_length, 1)]
    segmentations = [f'{CFG.check_video}/video_{CFG.save_syntax}/{i:09}.png' for i in range(0, dir_length, 1)]
    raft_model = initialize_raft_model()

    mean_tc = evaluate_temporal_consistency(raft_model, video_frames, segmentations, save_plot_name="video")
    print(f"video - Mean Temporal Consistency for video {CFG.plot_name}: ")
    for i, tc in enumerate(mean_tc):
        print(f"id {i}: {tc:.4f}")
    
    

    segmentations = [cv2.imread(f'{CFG.plot_name}/segmentation/{i:09}.png', cv2.IMREAD_GRAYSCALE) for i in range(0, dir_length, 60)]
    class_freq = {}
    array = np.array(segmentations)
    for i in range(10):
        class_freq[i] = np.sum(array == i)
    sum_freq = sum(class_freq.values())
    for i in range(10):
        class_freq[i] = class_freq[i] / sum_freq
    frame_score = np.load(f"plot/{CFG.save_syntax}_1/{CFG.plot_name}/frame.npy")
    video_score = np.load(f"plot/{CFG.save_syntax}_1/{CFG.plot_name}/video.npy")
    frame_norm = frame_score
    video_norm = video_score
    for i in range(frame_norm.shape[1]):
        plt.figure(figsize=(10, 5))
        plt.plot(frame_norm[:,i])
        plt.plot(video_norm[:,i])
        plt.legend(["frame", "video"])
        plt.savefig(f"plot/{CFG.save_syntax}_1/{CFG.plot_name}/norm_{i}.png")
        plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    for i in range(1, frame_norm.shape[1]):
        plt.scatter(frame_norm[:,i], video_norm[:,i], label=f"{i} - {class_freq[i]:.4f}", s=10)
    plt.legend([f"{i} - {class_freq[i]:.3f}" for i in range(frame_norm.shape[1])])
    plt.xlabel("frame")
    plt.ylabel("video")
    plt.plot([0,1], [0,1], color="black", linestyle="--")
    
    plt.subplot(122)
    frame_norm = np.nanmean(frame_norm, axis=1)
    video_norm = np.nanmean(video_norm, axis=1)
    print(f"frame_norm: {frame_norm}")
    print(f"video_norm: {video_norm}")
    plt.plot(frame_norm)
    plt.plot(video_norm)
    plt.legend(["frame", "video"])
    plt.savefig(f"plot/{CFG.save_syntax}_1/{CFG.plot_name}/norm.png")


if __name__ == "__main__":
    main()