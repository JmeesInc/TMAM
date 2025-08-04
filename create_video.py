import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from pathlib import Path
import gc
class CFG:
    save_syntax = "maxvit_tiny_tf_512"
    check_video = "/mnt/ssd1/EndoVis2022/train/video_36/"
    out_path = "../maxvit_tiny_tf_512_compare.mp4"

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
        else:
            truth = truth
        
        if include_frame_pred:
            frame_pred = cv2.imread(frame_pred_file)
            frame_pred = cv2.resize(frame_pred, (512, 512))
            frame_pred = cv2.applyColorMap((frame_pred*25).astype(np.uint8), cv2.COLORMAP_JET)
            write_frame = np.hstack([image, vid_pred_frame, frame_pred, truth])
        else:
            write_frame = np.hstack([image, vid_pred_frame, truth])
        
        out.write(write_frame)
        frame_counter += 1
        del write_frame, vid_pred_frame, frame_pred, image

        if frame_counter % 100 == 0:
            gc.collect()
        ###########################
        if frame_counter > 600:
            break
    out.release()

def main():
    create_compare_video(include_frame_pred=True)

if __name__ == "__main__":
    main()