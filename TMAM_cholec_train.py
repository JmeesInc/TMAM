import torch
import numpy as np
import os
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

def set_env():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["SAM2_BUILD_CUDA"] = "0"
    os.environ["SAM2_BUILD_ALLOW_ERRORS"] = "0"
    k = torch.randn(1)
    k = k.cuda()
    del k
    torch.cuda.empty_cache()
    gc.collect()

def convert_batch_norm_to_group_norm(model, num_groups=32):
    """BatchNorm2dをGroupNormに変換してバッチサイズ1に対応（より簡単な方法）"""
    import torch.nn as nn
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # GroupNormに変換
            new_module = nn.GroupNorm(
                num_groups=min(num_groups, module.num_features),
                num_channels=module.num_features,
                eps=module.eps,
                affine=module.affine
            )
            # 重みをコピー
            if module.affine:
                new_module.weight.data = module.weight.data.clone()
                new_module.bias.data = module.bias.data.clone()
            
            # 親モジュールで置き換え
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, new_module)
            else:
                setattr(model, child_name, new_module)
    return model

set_env()

from monai.losses import GeneralizedDiceFocalLoss

from base_model.alt_model import TimmSegModel_v3 as UNET
from util.converter2 import VideoSegModel2 as VideoSegModel
from util.data import TestDataset
import kornia.augmentation as K

import sys
sys.path.append("./sam2")

class CFG:
    backbone = "resnet101.a1h_in1k"
    save_syntax = "resnet101_bin"
    decoder = "deeplabv3"
    autocast = True
    image_size= 512
    train_csv = "cholecseg8k_train.csv"
    valid_csv = "cholecseg8k_test.csv"
    model_path = "base_model/weight/cholec/resnet101_bin.pth"
    num_epochs = 5
    batch_size = 1
    accum_steps = 8
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # マルチGPUを使用するかどうかのフラグ
    world_size = 1#torch.cuda.device_count()
    use_multi_gpu = False#True if torch.cuda.device_count() > 1 else False
    dist_url = 'tcp://localhost:12355'
    dist_backend = 'nccl'
    train_augmentation = K.AugmentationSequential(
        K.auto.TrivialAugment(),
        data_keys=["input", "mask"],
    )
    mask_num=13
    LABEL2CH = {
        0: 0,  50: 0, 255: 0, 
        5: 1,  11: 2, 12: 3, 13: 4,
        21: 5,  22: 6, 23: 7, 24: 8,
        25: 9,  31:10, 32:11, 33:12
    }
    NUM_CLASSES = 13


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
        mask = np.eye(CFG.NUM_CLASSES, dtype=np.float32)[idx_mask]

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
                    print(f"Saving video to {save_dir}")
                
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

    # 各クラスの平均DiceScoreを計算
    mean_dice_scores = []
    for class_idx in range(CFG.mask_num):
        if len(class_dice_scores[class_idx]) > 0:
            mean_dice = np.mean(class_dice_scores[class_idx])
            mean_dice_scores.append(mean_dice)
            print(f"Class {class_idx} | Dice Score: {mean_dice:.4f}")
        else:
            mean_dice_scores.append(0.0)
            print(f"Class {class_idx} | Dice Score: 0.0000 (no samples)")
    
    # 全体の平均DiceScore
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

def create_compare_video(out_path, video_dirs):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 512*3, 512
    out = cv2.VideoWriter(out_path, fourcc, 25, (w, h))

    image_files = []
    for video_dir in video_dirs:
        video_name = Path(video_dir).name
        # Assuming original data is in a structured path
        original_image_dir = f"/data4/shared/CholecSeg8k/{video_name}/"
        image_files.extend(sorted(glob(f"{original_image_dir}/*.png")))

    for image_file in tqdm(image_files, desc="Creating comparison video"):
        frame_name = Path(image_file).name
        video_name = Path(image_file).parent.name
        
        pred_dir = original_image_dir
        pred_file = f"{pred_dir}/{frame_name}"
        
        truth_file = image_file.replace(f"videos/{video_name}", f"annotations/{video_name}").replace(".png", "_watershed_mask.png")

        if not os.path.exists(pred_file) or not os.path.exists(truth_file):
            continue

        pred_mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        pred_mask = cv2.applyColorMap((pred_mask * 20).astype(np.uint8), cv2.COLORMAP_JET)

        image = cv2.imread(image_file)
        image = cv2.resize(image, (512, 512))
        
        truth_mask_raw = cv2.imread(truth_file, cv2.IMREAD_GRAYSCALE)
        truth_mask_idx = np.vectorize(CFG.LABEL2CH.get, otypes=[np.uint8])(truth_mask_raw)
        truth_mask = cv2.resize(truth_mask_idx, (512, 512), interpolation=cv2.INTER_NEAREST)
        truth_mask = cv2.applyColorMap((truth_mask * 20).astype(np.uint8), cv2.COLORMAP_JET)
        
        write_frame = np.hstack([image, pred_mask, truth_mask])
        
        out.write(write_frame)
    out.release()
    print(f"Comparison video saved to {out_path}")

from torch.nn import SyncBatchNorm

def replace_batch_norm_with_group_norm(model, num_groups=32):
    """BatchNorm2dをGroupNormに変換してバッチサイズ1に対応"""
    import torch.nn as nn
    
    def get_valid_num_groups(num_channels, target_groups):
        """チャンネル数がグループ数で割り切れるようにグループ数を調整"""
        if num_channels < target_groups:
            return 1
        # チャンネル数の約数を探す
        for i in range(min(target_groups, num_channels), 0, -1):
            if num_channels % i == 0:
                return i
        return 1  # デフォルトは1
    
    def replace_in_module(module):
        """再帰的にモジュール内のBatchNorm2dをGroupNormに置き換え"""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                # 有効なグループ数を計算
                valid_groups = get_valid_num_groups(child.num_features, num_groups)
                
                # GroupNormに変換
                new_module = nn.GroupNorm(
                    num_groups=valid_groups,
                    num_channels=child.num_features,
                    eps=child.eps,
                    affine=child.affine
                )
                # 重みをコピー
                if child.affine:
                    new_module.weight.data = child.weight.data.clone()
                    new_module.bias.data = child.bias.data.clone()
                
                # モジュールを置き換え
                setattr(module, name, new_module)
                print(f"Replaced BatchNorm2d with GroupNorm: {name} (channels: {child.num_features}, groups: {valid_groups})")
            else:
                # 再帰的に子モジュールを処理
                replace_in_module(child)
    
    replace_in_module(model)
    return model


def main():
    set_env()
    model = UNET(
        backbone=CFG.backbone,
        out_dim=CFG.mask_num,
        segtype=CFG.decoder
    )
    
    # BatchNorm2dをSyncBatchNormに変換してバッチサイズ1に対応
    model = replace_batch_norm_with_group_norm(model)
    
    model.eval()
    print(model(torch.randn(1, 3, 512, 512)).shape)
    if os.path.exists(CFG.model_path):
        print("Loading pre-trained weights...")
        model.load_state_dict(fix_key(torch.load(CFG.model_path)), strict=False)
    else:
        print("No pre-trained weights found, starting from scratch.")
    
    model.to(CFG.device)
    # データセットの準備
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.valid_csv)

    print(f"Train frames: {len(train_df)}")
    print(f"Val frames: {len(val_df)}")

    train_dataset = Dataset(train_df)
    val_dataset = Dataset(val_df)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

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
    
    
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")

    model2 = VideoSegModel(model.encoder, model.decoder, model.segmentation_head, device="cuda", sam2_predictor=predictor, index=[i for i in range(10)], depth=-1)

    if os.path.exists(f"base_model/weight/cholec/fine_tune_{CFG.save_syntax}.pth"):
        model2.load_state_dict(fix_key(torch.load(f"base_model/weight/cholec/fine_tune_{CFG.save_syntax}.pth")), strict=False)

    # Initialize optimizer and criterion
    optimizer = RAdamScheduleFree(model2.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    train_criterion = GeneralizedDiceFocalLoss(softmax=True)
    
    best_val_score = 0
    for epoch in range(CFG.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{CFG.num_epochs} ---")
        train(model2, train_loader, optimizer, train_criterion, CFG.device)
        
        # Configure model for validation (longer memory)
        #model2_val = VideoSegModel(model2.encoder, model2.decoder, model2.seg_head, device="cuda", sam2_predictor=predictor, index=[i for i in range(60)])
        val_loss = validate(model2, val_loader, valid_criterion, CFG.device, optimizer)
        print(f"Val Score: {val_loss:.4f}")
        
        if val_loss > best_val_score:
            best_val_score = val_loss
            torch.save(model2.state_dict(), f"base_model/weight/cholec/fine_tune_{CFG.save_syntax}_{epoch}.pth")
            print("Saved best model!")

    print(f"Best Val Score: {best_val_score:.4f}")

    val_video_dirs = val_df['file'].apply(lambda x: str(Path(x).parent.parent)).unique().tolist()
    #create_compare_video(
    #    out_path=f"/home/shunsuke/MICCAI2025/log/cholec_{CFG.save_syntax}_compare.mp4",
    #    video_dirs=val_video_dirs
    #)

if __name__ == "__main__":
    set_env()
    main()