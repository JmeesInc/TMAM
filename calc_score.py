import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import numpy as np
import cv2
from glob import glob
import tqdm
import gc
from monai.metrics import MeanIoU
from torchmetrics.segmentation import DiceScore
from monai.networks.utils import one_hot
class CFG:
    backbone = "tf_efficientnet_b7.ns_jft_in1k"
    save_syntax = "resnet101_bin"
    valid_dir = "/mnt/ssd1/EndoVis2022/train/"
    test_dir = "/mnt/ssd1/EndoVis2022/test/"
    device = "cpu"
    debug = False

def return_dirs(CFG):
    valid_txt = CFG.valid_dir + "/val_video.txt"
    with open(valid_txt) as f:
        val_dirs = f.readlines()
    
    val_dirs = [CFG.valid_dir + s.replace("\n", "") for s in val_dirs]
    if CFG.debug:
        val_dirs = val_dirs[:-4]
    return val_dirs

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

def recover_fast(tgt_image, lookup_array=color_to_class_array):
    # Use numpy advanced indexing instead of loops
    return lookup_array[tgt_image[..., 0], tgt_image[..., 1], tgt_image[..., 2]]

def rgb2onehot(pred):
    gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    class_indices = gray / 25
    class_indices = np.round(class_indices).astype(np.uint8)

def eval_iou(pred_dirs, video_pred = False, ablation = False):
    torch.cuda.empty_cache()
    gc.collect()
    Dice = DiceScore(num_classes=10, include_background=False, average=None, input_format="index")
    #Dice = DiceScore(num_classes=10, average="weighted", input_format="index").to(CFG.device)##################
    #if not video_pred:
    #    Dice = Dice.to(CFG.device)
    for pred_dir in pred_dirs:
        Dice_video = DiceScore(num_classes=10, include_background=False, average=None, input_format="index")
        target_files = sorted(glob(pred_dir + "/segmentation/*.png"))
        for target_file in tqdm.tqdm(target_files):
            if video_pred:
                pred_file = target_file.replace("/segmentation/", f"/video_{CFG.save_syntax}/")
            elif ablation:
                pred_file = target_file.replace("/segmentation/", f"/ablation_{CFG.save_syntax}/")
            else:
                pred_file = target_file.replace("/segmentation/", f"/{CFG.save_syntax}/")
            
            pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE).astype(np.int64)
            target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE).astype(np.int64)
            pred = cv2.resize(pred,(target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)

            #target, pred = (target!=0).astype(np.int64), (pred!=0).astype(np.int64)
            pred = one_hot(pred, 10)
            target = one_hot(target, 10)
            pred, target = torch.from_numpy(pred).to(CFG.device), torch.from_numpy(target).to(CFG.device)
            Dice.update(pred, target)
            Dice_video.update(pred, target)
        print(Dice_video.compute())
        Dice_video.reset()
    return Dice.compute()

def eval_iou2(pred_dirs, video_pred = False, ablation = False):
    metric = MeanIoU(include_background=False, reduction='mean', get_not_nans=False, ignore_empty=False)
    iou = []
    for pred_dir in pred_dirs:
        dir_iou = []
        target_files = sorted(glob(pred_dir + "/segmentation/*.png"))
        #dir_metric = MeanIoU(include_background=False, reduction='mean', get_not_nans=False, ignore_empty=False).to(CFG.device)
        for target_file in tqdm.tqdm(target_files):
            if video_pred:
                pred_file = target_file.replace("/segmentation/", f"/video_{CFG.save_syntax}/")
            elif ablation:
                pred_file = target_file.replace("/segmentation/", f"/ablation_{CFG.save_syntax}/")
            else:
                pred_file = target_file.replace("/segmentation/", f"/{CFG.save_syntax}/")

            pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE).astype(np.int64)
            target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE).astype(np.int64)
            pred = cv2.resize(pred,(target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
            target = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
            pred = one_hot(pred, 10, dim=1)
            target = one_hot(target, 10, dim=1)
            err = metric(pred, target)
            dir_iou.append(err)
            iou.append(err)
        print(np.nanmean(dir_iou))
    return np.nanmean(iou)
    
if __name__ == "__main__":
    val_dirs = return_dirs(CFG)
    print(val_dirs)
    print("Ablation study: ", eval_iou2(val_dirs, ablation=True))
    #print("Before module: ", eval_iou(val_dirs, video_pred=False))
    print("After module: ", eval_iou2(val_dirs, video_pred=True))
