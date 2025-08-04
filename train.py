import torch
import numpy as np
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

def set_env():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.backends.cudnn.benchmark = True

set_env()
from base_model.alt_model import TimmSegModel as UNET
#from segmentation_models_pytorch import DeepLabV3Plus

class CFG:
    backbone = "maxvit_tiny_tf_512.in1k"
    save_syntax = "maxvit_tiny_tf_512_ft"
    decoder = "unet"
    depth = 4
    autocast = True
    image_size=1024
    data_dir = "EndoVis2022/train/"
    num_epochs = 50
    batch_size = 8
    learning_rate = 2e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # マルチGPUを使用するかどうかのフラグ
    world_size = torch.cuda.device_count()
    use_multi_gpu = True if torch.cuda.device_count() > 1 else False
    dist_url = 'tcp://localhost:12355'
    dist_backend = 'nccl'
    train_augmentation = None
    model_path = "base_model/weight/best_model_maxvit_tiny_tf_512_2.pth"
    mask_num=10

class Dataset(BaseDataset):
    def __init__(self, dir_list, transform=None, train=False):
        self.dir_list = dir_list
        self.transform = transform
        self.image_paths = []   
        self.mask_paths = []

        for video_dir in self.dir_list:
            mask_paths = sorted(glob(f"{video_dir}/segmentation/*.png"))
            image_paths = [p.replace("segmentation", "rgb") for p in mask_paths]
            self.mask_paths.extend(mask_paths)
            self.image_paths.extend(image_paths)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx])[:, :, 0]

        # Resize
        image = cv2.resize(image, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_NEAREST)

        # One-hot encode mask
        if CFG.mask_num == 2:
            mask = (mask != 0).astype(np.int64)
            mask = np.eye(CFG.mask_num)[mask]
        else:
            mask = np.eye(CFG.mask_num)[mask]

        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

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
                # GPUメモリを解放
                #del images, masks, outputs
                #torch.cuda.empty_cache()
        
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
    val_dirs = loader.dataset.dir_list
    image_paths = loader.dataset.image_paths

    criterion.to(device)

    print(val_dirs)
    print(len(loader))
    print(len(val_dirs))
    print(image_paths[:2], image_paths[-2:])

    dir_counter = 0
    
    with torch.no_grad():
        with tqdm(loader, desc="Validation") as pbar:
            for i, (images, masks, video_start, flag) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                masks = masks.int().to(device, non_blocking=True)
                video_start = video_start.bool().to(device)
                outputs = model(images).softmax(dim=1)
                outputs = (outputs > 0.5).int()

                if video_start[0]:
                    save_dir = val_dirs[dir_counter] + f"/{CFG.save_syntax}"
                    os.makedirs(save_dir, exist_ok=True)
                    dir_counter += 1
                    if  i > 0:
                        val_loss.append(criterion.compute().cpu())

                if flag[0]:
                    criterion.update(outputs, masks)
                save_path = image_paths[i].replace("rgb", f"{CFG.save_syntax}")
                tgt = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy()
                tgt = tgt.astype(np.uint8)
                cv2.imwrite(save_path, tgt)

    val_loss.append(criterion.compute().cpu())
    print(val_loss)
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
    model = model.to(device)
    model = torch.compile(model)
    
    if CFG.use_multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # データセットの準備
    train_txt = CFG.data_dir + "/train_video.txt"
    valid_txt = CFG.data_dir + "/val_video.txt"
    with open(train_txt) as f:
        train_dirs = f.readlines()
    train_dirs = [CFG.data_dir + s.replace("\n", "") for s in train_dirs]
    train_dirs = train_dirs[1:]
    with open(valid_txt) as f:
        val_dirs = f.readlines()
    val_dirs = [CFG.data_dir + s.replace("\n", "") for s in val_dirs]
    
    print(train_dirs)
    train_dataset = Dataset(train_dirs, train=True)
    print(train_dataset.image_paths[:2], train_dataset.image_paths[-2:])
    val_dataset = Dataset(val_dirs)

    test_dataset = TestDataset(val_dirs, batch_size=1, eval=True)

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
    set_env()
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