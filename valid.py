import os
def set_env():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "3"
    os.environ["SAM2_BUILD_CUDA"] = "0"
    os.environ["SAM2_BUILD_ALLOW_ERRORS"] = "0"
set_env()

import numpy as np
import torch
import cv2
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torchmetrics.segmentation import MeanIoU
import gc

def set_env():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "3"
    os.environ["SAM2_BUILD_CUDA"] = "0"
    os.environ["SAM2_BUILD_ALLOW_ERRORS"] = "0"

set_env()

from monai.losses import DiceLoss
from train import Dataset as TrainDataset

from base_model.alt_model import TimmSegModel as UNET
from TMAM.util.model import VideoSegModel ###
from util.data import TestDataset

import sys
sys.path.append("./sam2")

class CFG:
    backbone = "maxvit_tiny_tf_512.in1k"
    save_syntax = "maxvit_tiny_tf_512_2"
    valid_dir = "/mnt/ssd1/EndoVis2022/train/"
    test_dir = "/mnt/ssd1/EndoVis2022/test/"
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    debug = False
    check_video = "/mnt/ssd1/EndoVis2022/train/video_36/"
    num_classes = 10
    model_path = "base_model/weight/best_model_maxvit_tiny_tf_512_2.pth"


def validate(model, loader, criterion, device):
    model.eval()
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

                if video_start[0]:
                    if i>0:
                        loss_value = criterion.compute()
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
                frame_counter = 0
                os.makedirs(save_dir, exist_ok=True)
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
        if k.startswith('_orig_mod.'):
            k = k[10:]
        if k== "memory":
            continue
        if k== "memory_pos":
            continue
        new_state_dict[k] = v
    return new_state_dict

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
        vid_pred_file = image_file.replace("rgb", f"video_{CFG.save_syntax}")
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
        out_dim=CFG.num_classes,
    )
    model.to(CFG.device)
    
    valid_txt = CFG.valid_dir + "/val_video.txt"
    with open(valid_txt) as f:
        val_dirs = f.readlines()
    
    val_dirs = [CFG.valid_dir + s.replace("\n", "") for s in val_dirs]

    val_dirs = val_dirs
    test_dirs = sorted(glob(CFG.test_dir + "/**/"))

    if CFG.debug:
        val_dirs = val_dirs[-2:]
        test_dirs = test_dirs[-2:]

    print(f"Val  dirs: {len(val_dirs)}, {val_dirs}")
    print(f"Test dirs: {len(test_dirs)}, {test_dirs}")

    val_dataset = TestDataset(val_dirs, batch_size=CFG.batch_size, eval=True, num_classes=CFG.num_classes)
    test_dataset = TestDataset(test_dirs, batch_size=CFG.batch_size, eval=False, num_classes=CFG.num_classes)

    print(f"Val  size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    
    valid_criterion = MeanIoU(num_classes=CFG.num_classes, per_class=True)
    
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda",)
    model2 = VideoSegModel(model.encoder, model.decoder, model.segmentation_head, device="cuda", sam2_predictor=predictor)
    model2.load_state_dict(fix_key(torch.load(
        CFG.model_path,
    )), strict=False)
    val_loss = validate(model2, val_loader, valid_criterion, CFG.device)
    test(model2, test_loader, CFG.device)
    
    print(f"Val Loss: {val_loss:.4f}")

    create_compare_video(include_frame_pred=True)

if __name__ == "__main__":
    set_env()
    main()