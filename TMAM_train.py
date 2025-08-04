import torch
import numpy as np
import os
import cv2
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torchmetrics.segmentation import MeanIoU
import gc
from schedulefree import RAdamScheduleFree

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

set_env()

from monai.losses import GeneralizedDiceFocalLoss
from train import Dataset as TrainDataset

from base_model.alt_model import TimmSegModel_v3 as UNET ###
from TMAM.util.model import TMAM
from util.data import TestDataset

import sys
sys.path.append("./sam2")

class CFG:
    backbone = "resnet101.a1h_in1k"
    save_syntax = "resnet101_bin"
    decoder = "deeplabv3"
    depth = 5
    autocast = True
    image_size=1024
    data_dir = "/mnt/ssd1/EndoVis2022/train/"
    valid_dir = "/mnt/ssd1/EndoVis2022/train/"
    test_dir = "/mnt/ssd1/EndoVis2022/test/"
    batch_size = 1
    accum_steps = 32
    learning_rate = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    debug = False
    check_video = "/mnt/ssd1/EndoVis2022/train/video_36/"
    model_path = "base_model/weight/ablation_resnet101_bin_2_0.4897.pth"

def train(model, loader, optimizer, criterion, device):
    model.train()
    turn_off_batchnorm(model)
    optimizer.train()
    epoch_loss = 0
    accumulation_steps = CFG.accum_steps
    scaler = torch.GradScaler(enabled=CFG.autocast)
    optimizer.zero_grad()
    with tqdm(loader, desc="Training") as pbar:
        for i, (images, masks, video_start, flag) in enumerate(loader):
            # Move tensors to device and process
            images = images.to(device)
            masks = masks.int().to(device)
            video_start = video_start.bool().to(device)
            
            if any(video_start):
                model.init_weights(video_start)
            outputs = model(images)  # .softmax(dim=1) if needed
            
            if flag[0]:
                outputs = F.interpolate(outputs, size=(1024, 1024), mode="bilinear")
                loss = criterion(outputs, masks)
                # Normalize loss to account for gradient accumulation
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * accumulation_steps
                pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")
            #else:
                # If flag is False, we effectively skip the step (but still need a value for tracking)
                #loss = torch.tensor(0.0, device=device)
            
            # Accumulate loss for progress tracking (scale back to original magnitude)
            #epoch_loss += loss.item() * accumulation_steps

            # Perform optimizer step every accumulation_steps or on the last batch
            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
            pbar.update(1)

def train_2d(model, loader, optimizer, criterion, device):
    model.train()
    turn_off_batchnorm(model)
    gc.collect()
    torch.cuda.empty_cache()
    optimizer.train()
    epoch_loss = 0
    accumulation_steps = CFG.accum_steps
    scaler = torch.GradScaler(enabled=CFG.autocast)
    optimizer.zero_grad()
    with tqdm(loader, desc="Training") as pbar:
        for i, (images, masks, video_start, flag) in enumerate(loader):
            # Move tensors to device and process
            images = images.to(device)
            masks = masks.int().to(device)

            outputs = model(images)  # .softmax(dim=1) if needed
            outputs = F.interpolate(outputs, size=(1024, 1024), mode="bilinear")
            
            if flag[0]:
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
    val_loss = []
    criterion.to(device)
    val_dirs = loader.dataset.dir_list
    
    video_counter = 0
    frame_counter = 0
    with torch.no_grad():
        for i, (images, masks, video_start, flag) in enumerate(tqdm(loader)):
            try:
                # Move tensors to device and process
                images = images.to(device)
                masks = masks.int().to(device)
                video_start = video_start.bool().to(device)
                
                if any(video_start):
                    model.init_weights(video_start)
                outputs = model(images).softmax(dim=1)
                outputs = F.interpolate(outputs, size=(1024, 1024), mode="bilinear")
                if video_start[0]:
                    if i>0:
                        loss_value = criterion.compute()
                        if loss_value.isnan().any():
                            val_loss.append(loss_value.cpu())
                        del loss_value
                        criterion.reset()
                    save_dir = val_dirs[video_counter] + f"/video_{CFG.save_syntax}/" ##########
                    frame_counter = 0
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Saving video to {save_dir}")
                    video_counter += 1
                
                file_name = f"{save_dir}/{frame_counter:09d}.png"
                frame_counter += 1

                if flag[0]:
                    criterion.update((outputs > 0.5).int(), masks)
                
                # Process and save output - optimize memory usage
                tgt = F.interpolate(outputs, size=(1080, 1920), mode="bilinear")
                del outputs
                tgt = torch.argmax(tgt.squeeze(), dim=0).cpu()
                tgt = tgt.numpy().astype(np.uint8)
                cv2.imwrite(file_name, tgt)

            finally:
                # Ensure cleanup happens even if an error occurs
                cleanup_tensors = [images, masks, video_start, flag]
                if 'tgt' in locals():
                    cleanup_tensors.append(tgt)
                
                for tensor in cleanup_tensors:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        del tensor
                torch.cuda.empty_cache()

    # Final validation loss computation with cleanup
    final_loss = criterion.compute()
    val_loss.append(final_loss.cpu())
    del final_loss
    criterion.reset()
    
    val_loss = torch.stack(val_loss).numpy()
    val_loss = np.mean(val_loss, axis=0)
    
    for i, scr in enumerate(val_loss):
        print(f"indice {i} | {float(scr):.4f}")
    
    return float(val_loss.mean())

def validate_2d(model, loader, criterion, device, optimizer=None):
    model.eval()
    if optimizer is not None:
        optimizer.eval()
    val_loss = []
    criterion.to(device)
    val_dirs = loader.dataset.dir_list
    
    video_counter = 0
    frame_counter = 0
    with torch.no_grad():
        for i, (images, masks, video_start, flag) in enumerate(tqdm(loader)):
            try:
                # Move tensors to device and process
                images = images.to(device)
                masks = masks.int().to(device)

                outputs = model(images).softmax(dim=1)
                outputs = F.interpolate(outputs, size=(1024, 1024), mode="bilinear")
                if video_start[0]:
                    if i>0:
                        loss_value = criterion.compute()
                        if loss_value.isnan().any():
                            val_loss.append(loss_value.cpu())
                        del loss_value
                        criterion.reset()
                    save_dir = val_dirs[video_counter] + f"/ablation_{CFG.save_syntax}/" ##########
                    frame_counter = 0
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Saving video to {save_dir}")
                    video_counter += 1
                
                file_name = f"{save_dir}/{frame_counter:09d}.png"
                frame_counter += 1

                if flag[0]:
                    criterion.update((outputs > 0.5).int(), masks)
                
                # Process and save output - optimize memory usage
                tgt = F.interpolate(outputs, size=(1080, 1920), mode="bilinear")
                del outputs
                tgt = torch.argmax(tgt.squeeze(), dim=0).cpu()
                tgt = tgt.numpy().astype(np.uint8)
                cv2.imwrite(file_name, tgt)

            finally:
                # Ensure cleanup happens even if an error occurs
                cleanup_tensors = [images, masks, video_start, flag]
                if 'tgt' in locals():
                    cleanup_tensors.append(tgt)
                
                for tensor in cleanup_tensors:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        del tensor
                torch.cuda.empty_cache()
    final_loss = criterion.compute()
    val_loss.append(final_loss.cpu())
    del final_loss
    criterion.reset()
    
    val_loss = torch.stack(val_loss).numpy()
    val_loss = np.mean(val_loss, axis=0)
    
    for i, scr in enumerate(val_loss):
        print(f"indice {i} | {float(scr):.4f}")
    
    return float(val_loss.mean())

def test(model, loader, device):
    model.eval()
    val_dirs = loader.dataset.dir_list
    
    video_counter = 0
    frame_counter = 0
    with torch.no_grad():
        for i, (images, video_start) in enumerate(tqdm(loader)):
            images = images.to(device)
            video_start = video_start.bool().to(device)
            model.init_weights(video_start)
            outputs = model(images).softmax(dim=1)

            if video_start[0]:
                save_dir = val_dirs[video_counter] + f"/video_{CFG.save_syntax}/"
                #frame_counter = 0
                #os.makedirs(save_dir, exist_ok=True)
                print(f"Saving to {save_dir}")
                video_counter += 1
            
            file_name = f"{save_dir}/{frame_counter:09d}.png"
            frame_counter += 1
            
            # Process and save output
            with torch.no_grad():
                tgt = F.interpolate(outputs, size=(1080, 1920), mode="bilinear")
                tgt = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy()
                tgt = tgt.astype(np.uint8)
                cv2.imwrite(file_name, tgt)
            
            # Explicit cleanup
            del images, video_start, outputs, tgt
            torch.cuda.empty_cache()
            
            # Periodic GPU cache clear
            if i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        elif k.startswith("_orig_mod."):
            k = k[10:]
        elif k=="memory" or k=="memory_pos":
            continue
        new_state_dict[k] = v
    return new_state_dict

def turn_off_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def create_compare_video(video_dir=CFG.check_video,
                          out_path = f"/home/shunsuke/MICCAI2025/log/{CFG.save_syntax}_compare.mp4",
                            include_frame_pred = False):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if include_frame_pred:
        w, h = 512*4, 512
    else:
        w, h = 512*3, 512
    out = cv2.VideoWriter(out_path, fourcc, 60, (w, h))

    # Get total number of frames
    image_files = sorted(glob(video_dir + f"/rgb/*.png"))
    total_frames = len(image_files)

    frame_counter = 0   
    for image_file in tqdm(image_files, desc="Creating comparison video"):
        vid_pred_file = image_file.replace("rgb", f"video_ft_{CFG.save_syntax}")
        frame_pred_file = image_file.replace("rgb", f"{CFG.save_syntax}")
        truth_file = image_file.replace("rgb", "segmentation")

        vid_pred_frame = cv2.imread(vid_pred_file)
        vid_pred_frame = cv2.resize(vid_pred_frame, (512, 512))
        vid_pred_frame = cv2.applyColorMap((vid_pred_frame*25).astype(np.uint8), cv2.COLORMAP_JET)
        
        image = cv2.imread(image_file)
        image = cv2.resize(image, (512, 512))
        
        if Path(truth_file).exists():
            truth = cv2.imread(truth_file)
            truth = cv2.resize(truth, (512, 512))
            truth = cv2.applyColorMap((truth*25).astype(np.uint8), cv2.COLORMAP_JET)
        
        if include_frame_pred:
            frame_pred = cv2.imread(frame_pred_file)
            frame_pred = cv2.resize(frame_pred, (512, 512))
            frame_pred = cv2.applyColorMap((frame_pred*25).astype(np.uint8), cv2.COLORMAP_JET)
            write_frame = np.hstack([image, vid_pred_frame, frame_pred, truth])
        else:
            write_frame = np.hstack([image, vid_pred_frame, truth])
        
        out.write(write_frame)
        frame_counter += 1
    out.release()


def main():
    set_env()
    model = UNET(
        backbone=CFG.backbone,
        segtype=CFG.decoder,
        out_dim=10,
    )
    model.load_state_dict(fix_key(torch.load(CFG.model_path)))
    model.to(CFG.device)
    #model = torch.compile(model)
    # データセットの準備
    train_txt = CFG.data_dir + "/train_video.txt"
    valid_txt = CFG.data_dir + "/val_video.txt"
    with open(train_txt) as f:
        train_dirs = f.readlines()
    train_dirs = [CFG.data_dir + s.replace("\n", "") for s in train_dirs]
    with open(valid_txt) as f:
        val_dirs = f.readlines()
    val_dirs = [CFG.data_dir + s.replace("\n", "") for s in val_dirs]
    
    test_dirs = sorted(glob(CFG.test_dir + "/**/"))

    if CFG.debug:
        train_dirs = train_dirs[1:]
        val_dirs = val_dirs[-2:]
        test_dirs = test_dirs[-2:]

    print(f"Val  dirs: {len(val_dirs)}, {val_dirs}")
    print(f"Test dirs: {len(test_dirs)}, {test_dirs}")

    train_dataset = TestDataset(train_dirs, batch_size=CFG.batch_size, train=True, eval=False)

    val_dataset = TestDataset(val_dirs, batch_size=CFG.batch_size, eval=True)
    test_dataset = TestDataset(test_dirs, batch_size=CFG.batch_size, eval=False)

    print(f"Val  size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )

    train_criterion = GeneralizedDiceFocalLoss(softmax=True)
    valid_criterion = MeanIoU(num_classes=10, per_class=True)
    val_loss = 0
    # ablation study
    optimizer2 = RAdamScheduleFree(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    #for epoch in range(3):
    #    train_2d(model, train_loader, optimizer2, train_criterion, CFG.device)
    #val_loss = validate_2d(model, val_loader, valid_criterion, CFG.device, optimizer2)
    #torch.save(model.state_dict(), f"base_model/weight/ablation_{CFG.save_syntax}_{epoch}_{val_loss:.4f}.pth")
    
    # reload model
    #model.load_state_dict(fix_key(torch.load(CFG.model_path)))
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda",)
    model2 = TMAM(model.encoder, model.decoder, model.segmentation_head, device="cuda", sam2_predictor=predictor, index=[i for i in range(10)], depth=CFG.depth)
            # Initialize optimizer and criterion
    #model2 = torch.compile(model2)
    optimizer = RAdamScheduleFree(model2.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    # Turn off barchnorm
    #model2.load_state_dict(fix_key(torch.load(CFG.model_path)), strict=False)
    for epoch in range(3):
        train(model2, train_loader, optimizer, train_criterion, CFG.device)
        torch.save(model2.state_dict(), f"base_model/weight/fine_tune_{CFG.save_syntax}_{epoch}_{val_loss:.4f}.pth")
        val_loss = validate(model2, val_loader, valid_criterion, CFG.device, optimizer)
        torch.save(model2.state_dict(), f"base_model/weight/fine_tune_{CFG.save_syntax}_{epoch}_{val_loss:.4f}.pth")
    
    #test(model2, test_loader, CFG.device)
    
    print(f"Val Loss: {val_loss:.4f}")

    #create_compare_video(include_frame_pred=True)

if __name__ == "__main__":
    main()