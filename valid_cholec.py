import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
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

from base_model.alt_model import TimmSegModel_v3 as UNET
#from segmentation_models_pytorch import DeepLabV3Plus

class CFG:
    backbone = "resnet101.a1h_in1k"
    save_syntax = "resnet101_bin"
    decoder = "deeplabv3"
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
    valid_csv = "cholecseg8k_test.csv"
    save_dir_root = f"/data4/shared/CholecSeg8k_save/img_{save_syntax}"

class CholecSegDataset(BaseDataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        mask_path = self.df['file'][idx]
        image_path = mask_path.replace('_watershed_mask', '')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        idx_mask = np.vectorize(CFG.LABEL2CH.get, otypes=[np.uint8])(mask_raw)
        mask = np.eye(CFG.NUM_CLASSES, dtype=np.float32)[idx_mask]
        image = cv2.resize(image, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, mask_path

def validate(model, loader, device, save_dir_root, num_classes=13):
    model.eval()
    dice = DiceScore(num_classes=num_classes, average=None, input_format="index")
    LABEL2CH = CFG.LABEL2CH
    with torch.no_grad():
        for images, masks, mask_paths in tqdm(loader):
            images = images.to(device)
            outputs = model(images).softmax(dim=1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # (B, H, W)
            masks = masks.cpu().numpy()  # (B, C, H, W)
            for i in range(images.size(0)):
                mask_path = mask_paths[i]
                video_name = os.path.basename(os.path.dirname(os.path.dirname(mask_path)))
                frame_name = os.path.basename(mask_path).replace('_watershed_mask.png', '.png')
                save_dir = os.path.join(save_dir_root, video_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, frame_name)
                cv2.imwrite(save_path, preds[i].astype(np.uint8))
                pred_map = np.vectorize(lambda x: LABEL2CH.get(x, 0), otypes=[np.uint8])(preds[i])
                target_map = np.vectorize(lambda x: LABEL2CH.get(x, 0), otypes=[np.uint8])(np.argmax(masks[i], axis=0))
                pred_tensor = torch.from_numpy(pred_map).long()
                target_tensor = torch.from_numpy(target_map).long()
                dice.update(pred_tensor, target_tensor)
    per_class = dice.compute().numpy()
    for i, d in enumerate(per_class):
        print(f"Class {i}: Dice = {d:.4f}")
    print(f"Mean Dice: {np.mean(per_class):.4f}")
    return per_class

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
    model.load_state_dict(fix_key(torch.load(CFG.model_path)))
    model = model.to(device)
    model = torch.compile(model)
    
    
    # データセットの準備
    train_df = pd.read_csv("/data4/src/shunsuke/MICCAI2025/cholecseg8k_train.csv")
    valid_df = pd.read_csv(CFG.valid_csv)

    val_dataset = CholecSegDataset(valid_df)

    train_sampler = None
    val_sampler = None
    num_workers = 32  # 単一GPU用のワーカー数も16に削減
    
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
    # Initialize optimizer and criterion
    from torchmetrics.segmentation import DiceScore
    valid_criterion = DiceScore(num_classes=CFG.mask_num, include_background=True, average=None)
    
    best_val_score = validate(model, val_loader, device, CFG.save_dir_root, num_classes=CFG.NUM_CLASSES)
    

if __name__ == "__main__":
    main()