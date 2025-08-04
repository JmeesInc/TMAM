import torch
import numpy as np
import cv2
from torchvision.transforms import functional as F
import sys
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from glob import glob
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--video_dir', type=str, required=True)
parser.add_argument('--save_plot_name', type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class CFG:
    save_syntax = args.model
    check_video = args.video_dir  # 例: video01
    plot_name = args.save_plot_name
    valid_dir = "/data4/shared/CholecSeg8k_save/"

# RAFTの初期化などはeval_tc.pyと同じ
sys.path.append('./RAFT/core')
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder

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

def compute_optical_flow(model, frame1, frame2):
    frame1 = load_image(frame1)
    frame2 = load_image(frame2)
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    with torch.no_grad():
        flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()

def warp_segmentation(segmentation, flow):
    h, w = segmentation.shape
    # flowのサイズに合わせてセグメンテーション画像をリサイズ
    flow_h, flow_w = flow.shape[:2]
    if (h, w) != (flow_h, flow_w):
        segmentation = cv2.resize(segmentation, (flow_w, flow_h), interpolation=cv2.INTER_NEAREST)
        h, w = flow_h, flow_w
    
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.stack(flow_map, axis=-1) + flow
    flow_map_y = flow_map[..., 1]
    flow_map_x = flow_map[..., 0]
    flow_map_x = np.clip(flow_map_x, 0, w - 1)
    flow_map_y = np.clip(flow_map_y, 0, h - 1)
    flow_map_x = flow_map_x.astype(np.float32)
    flow_map_y = flow_map_y.astype(np.float32)
    return cv2.remap(segmentation, flow_map_x, flow_map_y, interpolation=cv2.INTER_NEAREST)

def calculate_miou(seg1, seg2, num_classes=10):
    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(seg1 == cls, seg2 == cls).sum()
        union = np.logical_or(seg1 == cls, seg2 == cls).sum()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(np.nan)
    return np.array(ious) if ious else None

def evaluate_temporal_consistency(model, frame_files, seg_files, save_plot_name=None, num_classes=10):
    temporal_consistencies = []
    for i in tqdm(range(len(frame_files) - 1)):
        frame1, frame2 = frame_files[i], frame_files[i + 1]
        seg1, seg2 = seg_files[i], seg_files[i + 1]
        seg1 = cv2.imread(seg1, cv2.IMREAD_GRAYSCALE)
        seg2 = cv2.imread(seg2, cv2.IMREAD_GRAYSCALE)
        
        # 元のサイズを保持（後でリサイズするため）
        original_h, original_w = seg1.shape
        
        # RAFT用にフレーム画像を読み込み（元のサイズのまま）
        frame1_img = cv2.imread(frame1)
        frame2_img = cv2.imread(frame2)
        
        # optical flowを計算
        flow = compute_optical_flow(model, frame1, frame2)
        
        # セグメンテーション画像をflowのサイズに合わせてリサイズ
        flow_h, flow_w = flow.shape[:2]
        seg1_resized = cv2.resize(seg1, (flow_w, flow_h), interpolation=cv2.INTER_NEAREST)
        seg2_resized = cv2.resize(seg2, (flow_w, flow_h), interpolation=cv2.INTER_NEAREST)
        
        # warping
        warped_seg2 = warp_segmentation(seg1_resized, flow)
        
        # 結果を元のサイズに戻す
        warped_seg2 = cv2.resize(warped_seg2, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        seg2 = cv2.resize(seg2, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        temporal_consistencies.append(calculate_miou(warped_seg2, seg2, num_classes=num_classes))
    
    temporal_consistencies = np.array(temporal_consistencies)
    if save_plot_name is not None:
        mv_avg = []
        width = 10
        for i in range(temporal_consistencies.shape[0] - width):
            mv_avg.append(np.nanmean(temporal_consistencies[i:i+width,:], axis=0))
        plt.figure(figsize=(10, 5))
        plt.plot(mv_avg)
        plt.legend([f"{i}" for i in range(temporal_consistencies.shape[1])])
        os.makedirs(f"plot/{CFG.save_syntax}/{CFG.plot_name}", exist_ok=True)
        plt.savefig(f"plot/{CFG.save_syntax}/{CFG.plot_name}/{save_plot_name}.png")
        plt.close()
        np.save(f"plot/{CFG.save_syntax}/{CFG.plot_name}/{save_plot_name}.npy", np.array(mv_avg))
    return np.nanmean(temporal_consistencies, axis=0)

def main():
    # CSVから該当動画のデータを取得
    df = pd.read_csv("/data4/src/shunsuke/MICCAI2025/cholecseg8k_test.csv")
    video_df = df[df['group'] == CFG.check_video]
    
    if len(video_df) == 0:
        print(f"[ERROR] No data found for video {CFG.check_video} in CSV")
        return
    
    # 正解マスクパスとフレーム画像パス、推論画像パスを生成
    frame_files = []
    seg_files = []
    
    for _, row in video_df.iterrows():
        mask_path = row['file']  # 例: /data4/shared/CholecSeg8k/video01/video01_14939/frame_14939_endo_watershed_mask.png
        # フレーム画像パス（_watershed_maskを除去）
        frame_path = mask_path.replace('_watershed_mask.png', '.png')
        # 推論画像パス（valid_cholec.pyの保存構造に合わせる）
        # valid_cholec.py: save_dir_root = f"/data4/shared/CholecSeg8k_save/img_{save_syntax}"
        # 保存構造: /data4/shared/CholecSeg8k_save/img_{model}/video01/frame_14939_endo.png
        video_name = os.path.basename(os.path.dirname(os.path.dirname(mask_path)))  # video01
        frame_name = os.path.basename(mask_path).replace('_watershed_mask.png', '.png')  # frame_14939_endo.png
        pred_path = f"{CFG.valid_dir}/{CFG.save_syntax}/{video_name}/{frame_name}"
        
        if os.path.exists(frame_path) and os.path.exists(pred_path):
            frame_files.append(frame_path)
            seg_files.append(pred_path)
    
    if len(frame_files) == 0:
        print(f"[ERROR] No valid files found for video {CFG.check_video}")
        print(f"Expected frame files: {frame_path}")
        print(f"Expected pred files: {pred_path}")
        return
    
    print(f"Processing {len(frame_files)} frames for video {CFG.check_video}")
    
    raft_model = initialize_raft_model()
    mean_tc = evaluate_temporal_consistency(raft_model, frame_files, seg_files, save_plot_name="video", num_classes=10)
    print(f"video - Mean Temporal Consistency for video {CFG.plot_name}: ")
    for i, tc in enumerate(mean_tc):
        print(f"id {i}: {tc:.4f}")

if __name__ == "__main__":
    main() 