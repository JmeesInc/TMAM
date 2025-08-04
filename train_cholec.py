import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from glob import glob
from tqdm import tqdm
import pandas as pd
from torchmetrics.segmentation import DiceScore
import segmentation_models_pytorch as smp

from monai.losses import GeneralizedDiceFocalLoss
from schedulefree import RAdamScheduleFree

from util import vis
from util.data import TestDataset

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from collections import OrderedDict
from torchvision.transforms.v2 import TrivialAugmentWide

def set_env():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)

set_env()

class CFG:
    backbone = "tu-resnet101"
    save_syntax = "resnet101-deeplabv3-cholec"
    decoder = "deeplabv3"
    depth = 4
    autocast = True
    image_size=512
    
    num_epochs = 50
    mask_num=13
    batch_size = 32
    learning_rate = 1e-4
    train_augmentation = TrivialAugmentWide()

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

        mask_onehot = np.eye(CFG.mask_num, dtype=np.float32)[idx_mask].transpose(2, 0, 1)

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
    scaler = torch.GradScaler(enabled=CFG.autocast)
    
    with tqdm(loader, desc="Training") as pbar:
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.autocast("cuda", dtype=torch.float16, enabled=CFG.autocast):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
            del all_images, all_outputs, all_targets
        
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
    
    torch.manual_seed(42)
    np.random.seed(42)
    

    model = smp.DeepLabV3(
        encoder_name=CFG.backbone, classes=CFG.mask_num
    )
    model = model.to(device)
    
    if CFG.use_multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    train_df = pd.read_csv("cholecseg8k_train.csv")
    valid_df = pd.read_csv("cholecseg8k_test.csv")

    train_dataset = CholecSegDataset(train_df, transform=CFG.train_augmentation)
    val_dataset = CholecSegDataset(valid_df, transform=None)
    test_dataset = CholecSegDataset(valid_df, transform=None)

    # DataLoaderの設定
    if CFG.use_multi_gpu:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        num_workers = 16 // world_size 
    else:
        train_sampler = None
        val_sampler = None
        num_workers = 32 
    
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
    
    optimizer = RAdamScheduleFree(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    train_criterion = GeneralizedDiceFocalLoss(softmax=True)
    valid_criterion = DiceScore(num_classes=CFG.mask_num, include_background=True, average=None)
    
    best_val_score = 0
    for epoch in range(CFG.num_epochs):
        if CFG.use_multi_gpu:
            train_sampler.set_epoch(epoch) 
            
        print(f"\nEpoch {epoch+1}/{CFG.num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, train_criterion, device)
        val_score = validate(model, val_loader, train_criterion, valid_criterion, device, epoch, save_vid=False, optimizer=optimizer)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_score:.4f}")
        
        if val_score > best_val_score:
            best_val_score = val_score
            if CFG.use_multi_gpu:
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
    
    if CFG.use_multi_gpu:
        model.module.load_state_dict(torch.load(CFG.model_path))
    else:
        model.load_state_dict(torch.load(CFG.model_path))

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