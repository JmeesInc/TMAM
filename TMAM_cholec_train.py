import os
def set_env():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["SAM2_BUILD_CUDA"] = "0"
    os.environ["SAM2_BUILD_ALLOW_ERRORS"] = "0"
set_env()

import torch
import numpy as np
import cv2
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.optim as optim
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torchmetrics.segmentation import DiceScore
import gc
from schedulefree import RAdamScheduleFree
import pandas as pd
import torch.nn.functional as F

import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
from monai.losses import GeneralizedDiceFocalLoss

import segmentation_models_pytorch as smp
from util.model import TMAM, replace_batch_norm_with_group_norm
import kornia.augmentation as K
from torchvision.transforms.v2 import TrivialAugmentWide

import sys
sys.path.append("./sam2")

class CFG:
    backbone = "tu-resnet101"
    save_syntax = "resnet101-deeplabv3-cholec"
    decoder = "deeplabv3"
    depth = 4
    autocast = True
    image_size=512
    
    num_epochs = 50
    mask_num=13
    batch_size = 1
    learning_rate = 1e-4
    train_augmentation = TrivialAugmentWide()

    train_csv = "cholecseg8k_train.csv"
    valid_csv = "cholecseg8k_test.csv"
    load_model_path = f"base_model/weight/{save_syntax}.pth"
    save_model_path = f"base_model/weight/TMAM_{save_syntax}.pth"
    num_epochs = 3
    accum_steps = 8
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # マルチGPUを使用するかどうかのフラグ
    world_size = torch.cuda.device_count()
    use_multi_gpu = True if torch.cuda.device_count() > 1 else False
    dist_url = 'tcp://localhost:12355'
    dist_backend = 'nccl'

    model_path = f"base_model/weight/{save_syntax}.pth"
    LABEL2CH = {
        0: 0,  50: 0, 255: 0, 
        5: 1,  11: 2, 12: 3, 13: 4,
        21: 5,  22: 6, 23: 7, 24: 8,
        25: 9,  31:10, 32:11, 33:12
    }


class Dataset(BaseDataset):
    def __init__(self, df, transform=None):
        self.df = df.sort_values('file').reset_index(drop=True)
        self.transform = transform
        
        # Extract video identifier and determine video start frames
        self.df['video'] = self.df['file'].apply(lambda x: Path(x).parent.parent.name)
        self.video_starts = self.df['video'] != self.df['video'].shift(1)
        
        self.image_paths = self.df['file'].str.replace('_watershed_mask', '').tolist()
        self.mask_paths = self.df['file'].tolist()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        idx_mask = np.vectorize(CFG.LABEL2CH.get, otypes=[np.uint8])(mask_raw)
        mask = np.eye(CFG.mask_num, dtype=np.float32)[idx_mask]

        video_start = self.video_starts.iloc[idx]

        # Resize
        image = cv2.resize(image, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, video_start, mask_path

def train(model, loader, optimizer, criterion, device):
    model.train()
    optimizer.train()
    epoch_loss = 0
    accumulation_steps = CFG.accum_steps
    scaler = torch.GradScaler(enabled=CFG.autocast)
    optimizer.zero_grad()
    with tqdm(loader, desc="Training") as pbar:
        for i, (images, masks, video_start, mask_paths) in enumerate(loader):
            # Move tensors to device and process
            images = images.to(device)
            masks = masks.to(device) # Keep as float for loss, but argmax for metric
            video_start = video_start.bool().to(device)
            
            if any(video_start):
                model.init_weights(video_start)
            outputs = model(images).softmax(dim=1)
            
            loss = criterion(outputs, masks)
            # Normalize loss to account for gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")
            
            # Perform optimizer step every accumulation_steps or on the last batch
            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            pbar.update(1)

def validate(model, loader, criterion, device, optimizer=None):
    model.eval()
    if optimizer is not None:
        optimizer.eval()
    
    # 各クラスごとのDiceScoreを累積するためのリスト
    class_dice_scores = [[] for _ in range(CFG.mask_num)]
    
    with torch.no_grad():
        for i, (images, masks, video_start, mask_paths) in enumerate(tqdm(loader)):
            try:
                # Move tensors to device and process
                images = images.to(device)
                masks = masks.to(device) # Keep as float for loss, but argmax for metric
                video_start = video_start.bool().to(device)
                
                if any(video_start):
                    model.init_weights(video_start)
                outputs = model(images).softmax(dim=1)

                if video_start[0]:
                    video_name = Path(mask_paths[0]).parent.parent.name
                    save_dir = f"/data4/shared/CholecSeg8k_save/{CFG.save_syntax}/{video_name}/"
                    os.makedirs(save_dir, exist_ok=True)
                
                frame_name = Path(mask_paths[0]).name.replace('_watershed_mask.png', '.png')
                file_name = f"{save_dir}/{frame_name}"

                # 各クラスごとのDiceScoreを計算
                pred = outputs.argmax(1)  # [B, H, W]
                target = masks.argmax(1)  # [B, H, W]
                
                # 各クラスについてDiceScoreを計算
                for class_idx in range(CFG.mask_num):
                    pred_class = (pred == class_idx).float()
                    target_class = (target == class_idx).float()
                    
                    # Dice coefficient計算
                    intersection = (pred_class * target_class).sum()
                    union = pred_class.sum() + target_class.sum()
                    
                    if union > 0:
                        dice_score = (2.0 * intersection) / union
                    else:
                        dice_score = torch.tensor(1.0 if intersection == 0 else 0.0, device=device)
                    
                    class_dice_scores[class_idx].append(dice_score.cpu().item())
                
                # Process and save output - optimize memory usage
                tgt = F.interpolate(outputs, size=(512, 512), mode="bilinear")
                del outputs
                tgt = torch.argmax(tgt.squeeze(), dim=0).cpu().detach()
                tgt = tgt.numpy().astype(np.uint8)
                # Ensure image dimensions are valid for PNG format
                if tgt.shape[0] > 0 and tgt.shape[1] > 0 and tgt.shape[0] <= 65535 and tgt.shape[1] <= 65535:
                    cv2.imwrite(file_name, tgt)
                else:
                    print(f"Warning: Invalid image dimensions {tgt.shape} for {file_name}")

            finally:
                # Ensure cleanup happens even if an error occurs
                cleanup_tensors = [images, masks, video_start]
                if 'tgt' in locals():
                    del tgt # No need to append to list, just delete
                
                for tensor in cleanup_tensors:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        del tensor
                torch.cuda.empty_cache()
            
            # More aggressive periodic cleanup
            if i % 1000 == 0 and i > 0:
                torch.cuda.empty_cache()
                gc.collect()

    mean_dice_scores = []
    for class_idx in range(CFG.mask_num):
        if len(class_dice_scores[class_idx]) > 0:
            mean_dice = np.mean(class_dice_scores[class_idx])
            mean_dice_scores.append(mean_dice)
            print(f"Class {class_idx} | Dice Score: {mean_dice:.4f}")
        else:
            mean_dice_scores.append(0.0)
            print(f"Class {class_idx} | Dice Score: 0.0000 (no samples)")
    
    overall_mean_dice = np.mean(mean_dice_scores)
    print(f"Overall Mean Dice Score: {overall_mean_dice:.4f}")
    
    return overall_mean_dice

import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def main():
    set_env()    
    model = smp.DeepLabV3Plus(
        encoder_name=CFG.backbone,
        classes=CFG.mask_num
    )
    model.load_state_dict(torch.load(CFG.load_model_path), strict=False)
    model.eval()
    
    model = replace_batch_norm_with_group_norm(model) # Since TMAM is trained with batch size 1, we need to replace batch norm with group norm
    model.to(CFG.device)
    
    train_df = pd.read_csv("cholecseg8k_train.csv")
    valid_df = pd.read_csv("cholecseg8k_test.csv")

    train_dataset = Dataset(train_df)
    val_dataset = Dataset(valid_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    
    valid_criterion = DiceScore(num_classes=CFG.mask_num, average='none', include_background=False)

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", 
        "sam2/checkpoints/sam2.1_hiera_large.pt", 
        device="cuda")

    model2 = TMAM(model.encoder, model.decoder, model.segmentation_head, device="cuda", sam2_predictor=predictor, index=[i for i in range(10)], depth=-1)

    if os.path.exists(f"base_model/weight/cholec/fine_tune_{CFG.save_syntax}.pth"):
        model2.load_state_dict(fix_key(torch.load(f"base_model/weight/cholec/fine_tune_{CFG.save_syntax}.pth")), strict=False)

    optimizer = RAdamScheduleFree(model2.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    train_criterion = GeneralizedDiceFocalLoss(softmax=True)
    
    best_val_score = 0
    for epoch in range(CFG.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{CFG.num_epochs} ---")
        train(model2, train_loader, optimizer, train_criterion, CFG.device)
        val_loss = validate(model2, val_loader, valid_criterion, CFG.device, optimizer)
        print(f"Val Score: {val_loss:.4f}")
        
        if val_loss > best_val_score:
            best_val_score = val_loss
            torch.save(model2.state_dict(), f"base_model/weight/cholec/fine_tune_{CFG.save_syntax}_{epoch}.pth")
            print("Saved best model!")

    print(f"Best Val Score: {best_val_score:.4f}")

if __name__ == "__main__":
    set_env()
    main()