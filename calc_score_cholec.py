import os
import numpy as np
import cv2
from glob import glob
import tqdm
import gc
import pandas as pd
import torch

from torchmetrics.segmentation import DiceScore

class CFG:
    save_syntax = "maxvit_tiny_tf_512" 
    data_dir = "/data4/shared/CholecSeg8k/"
    valid_dir = "/data4/shared/CholecSeg8k_save/"
    valid_csv = "cholecseg8k_test.csv"
    device = "cpu"
    debug = False

def return_dirs(CFG):
    df = pd.read_csv(CFG.valid_csv)
    # 正解マスク
    image_files = df['file'].tolist()
    # 推論結果
    pred_files = []
    for file in image_files:
        # 例: /data4/shared/CholecSeg8k/video01/frame_00001_watershed_mask.png
        # 推論結果: /data4/shared/CholecSeg8k_save/resnet101_bin/video01/frame_00001.png
        video_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
        frame_name = os.path.basename(file).replace('_watershed_mask.png', '.png')
        pred_file = os.path.join(CFG.valid_dir, CFG.save_syntax, video_name, frame_name)
        pred_files.append(pred_file)
    print(len(image_files), len(pred_files))
    return image_files, pred_files

def calc_dice(file):
    target_file, pred_file = file
    pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
    if pred is None or target is None:
        return 0.0
    pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    pred = (pred != 0).astype(np.int64)
    target = (target != 0).astype(np.int64)
    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum + target_sum == 0:
        return 1.0
    dice = (2.0 * intersection) / (pred_sum + target_sum)
    return dice

def calc_dice_per_class(file, num_classes=13):
    target_file, pred_file = file
    pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
    if pred is None or target is None:
        return [0.0] * num_classes
    pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    # Cholecのラベルマッピング
    # 0, 50, 255→0, 5→1, 11→2, ...
    LABEL2CH = {
        0: 0,  50: 0, 255: 0, 
        5: 1,  11: 2, 12: 3, 13: 4,
        21: 5,  22: 6, 23: 7, 24: 8,
        25: 9,  31:10, 32:11, 33:12
    }
    pred = np.vectorize(LABEL2CH.get, otypes=[np.uint8])(pred)
    target = np.vectorize(LABEL2CH.get, otypes=[np.uint8])(target)
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.int32)
        target_cls = (target == cls).astype(np.int32)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        pred_sum = pred_cls.sum()
        target_sum = target_cls.sum()
        if pred_sum + target_sum == 0:
            dices.append(1.0)
        else:
            dices.append((2.0 * intersection) / (pred_sum + target_sum))
    return dices

from tqdm.contrib.concurrent import process_map

def eval_iou(target_files, pred_files):
    Dice = process_map(calc_dice, zip(target_files, pred_files), max_workers=10, total=len(target_files))
    return np.mean(Dice)

def eval_iou_per_class_torchmetrics(target_files, pred_files, num_classes=13):
    dice = DiceScore(num_classes=num_classes, average=None, input_format="index")
    LABEL2CH = {
        0: 0,  50: 0, 255: 0, 
        5: 1,  11: 2, 12: 3, 13: 4,
        21: 5,  22: 6, 23: 7, 24: 8,
        25: 9,  31:10, 32:11, 33:12
    }
    for target_file, pred_file in tqdm.tqdm(zip(target_files, pred_files), total=len(target_files)):
        pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
        if pred is None or target is None:
            continue
        pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        # ラベルマッピング
        pred = np.vectorize(lambda x: LABEL2CH.get(x, 0), otypes=[np.uint8])(pred)
        target = np.vectorize(lambda x: LABEL2CH.get(x, 0), otypes=[np.uint8])(target)
        pred = torch.from_numpy(pred).long()
        target = torch.from_numpy(target).long()
        dice.update(pred, target)
    per_class = dice.compute().numpy()
    for i, d in enumerate(per_class):
        print(f"Class {i}: Dice = {d:.4f}")
    print(f"Mean Dice: {np.mean(per_class):.4f}")
    return per_class

if __name__ == "__main__":
    print("save_syntax: ", CFG.save_syntax)
    image_files, pred_files = return_dirs(CFG)
    print("Dice Score (mean): ", eval_iou(image_files, pred_files))
    print("--- Per-class Dice (torchmetrics) ---")
    eval_iou_per_class_torchmetrics(image_files, pred_files, num_classes=13)