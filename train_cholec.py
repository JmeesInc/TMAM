import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import torch
torch.backends.cudnn.benchmark = True
import torch
import numpy as np
import pandas as pd
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torchmetrics.segmentation import MeanIoU, DiceScore

from monai.losses import GeneralizedDiceFocalLoss
from schedulefree import RAdamScheduleFree
#from monai.metrics import MeanIoU
import kornia.augmentation as K

from util import vis
from util.data import TestDataset

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from collections import OrderedDict

from base_model.alt_model import TimmSegModel as UNET
#from segmentation_models_pytorch import DeepLabV3Plus

class CFG:
    backbone = "tf_efficientnet_b7.ns_jft_in1k"
    save_syntax = "tf_efficientnet_b7"
    decoder = "unet"
    depth = 4
    autocast = True
    image_size=512
    num_epochs = 25
    batch_size = 32
    learning_rate = 2e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # マルチGPUを使用するかどうかのフラグ
    world_size = torch.cuda.device_count()
    use_multi_gpu = True if torch.cuda.device_count() > 1 else False
    dist_url = 'tcp://localhost:12355'
    dist_backend = 'nccl'
    train_augmentation = None
    model_path = f"base_model/weight/cholec/{save_syntax}.pth"
    mask_num=13
    LABEL2CH = {
        0: 0,  50: 0, 255: 0, 
        5: 1,  11: 2, 12: 3, 13: 4,
        21: 5,  22: 6, 23: 7, 24: 8,
        25: 9,  31:10, 32:11, 33:12
    }
    NUM_CLASSES = 13

class CholecSegDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mask_path  = self.df['file'][idx]
        image_path = mask_path.replace('_watershed_mask', '')
        image      = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask_raw   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        idx_mask = np.vectorize(CFG.LABEL2CH.get, otypes=[np.uint8])(mask_raw)

        mask_onehot = np.eye(CFG.NUM_CLASSES, dtype=np.float32)[idx_mask].transpose(2, 0, 1)

        image = cv2.resize(image, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask_onehot = torch.from_numpy(
            cv2.resize(mask_onehot.transpose(1,2,0), (CFG.image_size, CFG.image_size),
                       interpolation=cv2.INTER_NEAREST)
        ).permute(2,0,1)

        if self.transform is not None:
            image, mask_onehot = self.transform(image, mask_onehot)

        return image, mask_onehot


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    optimizer.train()
    epoch_loss = 0
    #augmentation = CFG.train_augmentation.to(device)
    scaler = torch.GradScaler(enabled=CFG.autocast)
    
    with tqdm(loader, desc="Training") as pbar:
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            #augmented = augmentation(images, masks)
            #images, masks = augmented[0], augmented[1]
            
            optimizer.zero_grad()
            
            with torch.autocast("cuda", dtype=torch.float16, enabled=CFG.autocast):
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return epoch_loss / len(loader)

def validate(model, loader, train_criterion, criterion, device, epoch, save_vid=False, optimizer=None):
    if optimizer is not None:
        optimizer.eval()
    model.eval()
    val_loss = 0
    train_criterion.to(device)
    scaler = torch.GradScaler("cuda", enabled=CFG.autocast)
    
    # 必要な場合のみリストを初期化
    all_images = [] if save_vid else None
    all_outputs = [] if save_vid else None
    all_targets = [] if save_vid else None
    
    with torch.no_grad():
        with tqdm(loader, desc="Validation") as pbar:
            for images, masks in pbar:
                images = images.to(device, non_blocking=True)
                masks = masks.int().to(device, non_blocking=True)
                
                with torch.autocast("cuda", dtype=torch.float16, enabled=CFG.autocast):
                    outputs = model(images).softmax(dim=1)
                    outputs = (outputs > 0.5).int()
                    #loss = train_criterion(outputs, masks)
                
                #val_loss += loss
                #pbar.set_postfix(loss=f"{loss.mean():.4f}")
                
                # Store all batches
                if save_vid:
                    for i in range(images.size(0)):
                        img = images[i].cpu().transpose(0,1).transpose(1,2).numpy()
                        out = outputs[i].cpu()
                        out = torch.argmax(out, dim=0).numpy()
                        out = cv2.applyColorMap((out*25).astype(np.uint8), cv2.COLORMAP_JET)  # Apply colormap
                        tgt = masks[i].cpu()
                        tgt = torch.argmax(tgt, dim=0).numpy()
                        tgt = cv2.applyColorMap((tgt*25).astype(np.uint8), cv2.COLORMAP_JET)  # Apply colormap
                        
                        if i<1000:
                            all_images.append(img)
                            all_outputs.append(out)
                            all_targets.append(tgt)
                
                criterion.update(outputs.cpu(), masks.cpu())
        
        if save_vid:
            vis.save_video(all_images, all_outputs, all_targets, f"log/conv_epoch_{epoch}.mp4")
            # メモリ解放
            del all_images, all_outputs, all_targets
        
        #val_loss /= len(loader)
        val_loss = criterion.compute()
        for i, scr in enumerate(val_loss):
            print(f"indice {i} | {float(scr):.4f}")
    return val_loss.mean().item()

def test(model, loader, criterion, device):
    model.eval()
    val_loss = []

    criterion.to(device)

    with torch.no_grad():
        with tqdm(loader, desc="Validation") as pbar:
            for i, (images, masks, video_start, flag) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                masks = masks.int().to(device, non_blocking=True)
                video_start = video_start.bool().to(device)
                outputs = model(images).softmax(dim=1)
                outputs = (outputs > 0.5).int()

                if video_start[0]:
                    if i > 0:
                        val_loss.append(criterion.compute().cpu())

                if flag[0]:
                    criterion.update(outputs, masks)
                tgt = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy()
                tgt = tgt.astype(np.uint8)

    val_loss.append(criterion.compute().cpu())
    val_loss = np.array(val_loss).mean(0)
    for i, scr in enumerate(val_loss):
        print(f"indice {i} | {float(scr):.4f}")
    return val_loss.mean().item()

def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
    if CFG.use_multi_gpu:
        dist.init_process_group(
            backend=CFG.dist_backend,
            init_method=CFG.dist_url,
            world_size=world_size,
            rank=rank
        )

def cleanup_ddp():
    """Cleanup DDP process group"""
    if CFG.use_multi_gpu:
        dist.destroy_process_group()

def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        elif k.startswith('_orig_mod.'):
            k = k[10:]
        new_state_dict[k] = v
    return new_state_dict

def main(rank=0, world_size=1):
    if CFG.use_multi_gpu:
        setup_ddp(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = CFG.device
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    

    if CFG.decoder == "deeplabv3":
        model = UNET(
            backbone=CFG.backbone,
            out_dim=CFG.mask_num,
            segtype = CFG.decoder
        )
    else:
        model = UNET(
            backbone=CFG.backbone,
            out_dim=CFG.mask_num,
        )
    model.load_state_dict(fix_key(torch.load(CFG.model_path)))#########
    model = model.to(device)
    model = torch.compile(model)
    
    if CFG.use_multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # データセットの準備
    train_df = pd.read_csv("/data4/src/shunsuke/MICCAI2025/cholecseg8k_train.csv")
    valid_df = pd.read_csv("/data4/src/shunsuke/MICCAI2025/cholecseg8k_test.csv")

    train_dataset = CholecSegDataset(train_df)
    val_dataset = CholecSegDataset(valid_df)
    test_dataset = CholecSegDataset(valid_df)

    # DataLoaderの設定
    if CFG.use_multi_gpu:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        num_workers = 16 // world_size  # ワーカー数を64から16に削減
    else:
        train_sampler = None
        val_sampler = None
        num_workers = 32  # 単一GPU用のワーカー数も16に削減
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize optimizer and criterion
    from torchmetrics.segmentation import DiceScore
    optimizer = RAdamScheduleFree(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    train_criterion = GeneralizedDiceFocalLoss(softmax=True)
    valid_criterion = DiceScore(num_classes=CFG.mask_num, include_background=True, average=None)
    ##scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=CFG.num_epochs, eta_min=1e-6
    #)
    
    # Training loop
    epoch = 0
    best_val_score = validate(model, val_loader, train_criterion, valid_criterion, device, epoch, save_vid=False, optimizer=optimizer)
    for epoch in range(CFG.num_epochs):
        if CFG.use_multi_gpu:
            train_sampler.set_epoch(epoch)  # エポックごとにシャッフル
            
        print(f"\nEpoch {epoch+1}/{CFG.num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, train_criterion, device)
        val_score = validate(model, val_loader, train_criterion, valid_criterion, device, epoch, save_vid=False, optimizer=optimizer)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_score:.4f}")
        
        #scheduler.step(val_score)
        
        if val_score > best_val_score:
            best_val_score = val_score
            if CFG.use_multi_gpu:
                # When using DDP, the compiled model is wrapped as model.module.
                # Check if the compiled model holds the original model in _orig_mod.
                if hasattr(model.module, "_orig_mod"):
                    state_dict = model.module._orig_mod.state_dict()
                else:
                    state_dict = model.module.state_dict()
            else:
                if hasattr(model, "_orig_mod"):
                    state_dict = model._orig_mod.state_dict()
                else:
                    state_dict = model.state_dict()
            torch.save(state_dict, CFG.model_path)
            print("Saved best model!")
    
    # Load best model
    if CFG.use_multi_gpu:
        model.module.load_state_dict(torch.load(CFG.model_path))
    else:
        model.load_state_dict(torch.load(CFG.model_path))

    # Test
    test(model, test_loader, valid_criterion, device)

    cleanup_ddp()

if __name__ == "__main__":
    if CFG.use_multi_gpu:
        import torch.multiprocessing as mp
        mp.spawn(
            main,
            args=(CFG.world_size,),
            nprocs=CFG.world_size,
            join=True
        )
    else:
        main()