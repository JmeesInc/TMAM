import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        #img = img.detach()
        #img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def save(imgs, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        #img = img.detach()
        #img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(path)

# save sequence of fix as an video
import cv2
def save_video(images, targets, truths, path, fps=1):
    h, w = images[0].shape[:2]
    w *= 3
    # Change codec to mp4v and ensure path ends with .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for img, target, truth in zip(images, targets, truths):
        # Convert to uint8 if not already
        img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        target = target.astype(np.uint8) if target.dtype != np.uint8 else target 
        truth = truth.astype(np.uint8) if truth.dtype != np.uint8 else truth

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR) 
        #truth = cv2.cvtColor(truth, cv2.COLOR_RGB2BGR)
        
        # Stack and ensure uint8 type
        stacked = np.hstack([img, target, truth]).astype(np.uint8)
        out.write(stacked)
    out.release()

def save_video2(images, targets, path, fps=1):
    h, w = images[0].shape[:2]
    w *= 2
    # Change codec to mp4v and ensure path ends with .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for img, target in zip(images, targets):
        # Convert to uint8 if not already
        img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        target = target.astype(np.uint8) if target.dtype != np.uint8 else target 

        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR) 
        
        # Stack and ensure uint8 type
        stacked = np.hstack([img, target]).astype(np.uint8)
        out.write(stacked)
    out.release()

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def show_video(images, targets, truths, fps=20, repeat_delay=1000):
    """Display video in jupyter notebook using matplotlib animation"""
    fig = plt.figure(figsize=(15, 5))
    frames = []
    
    for i in range(len(images)):
        stacked = np.hstack([images[i], targets[i], truths[i]])
        frames.append([plt.imshow(stacked, animated=True)])
        plt.xticks([])
        plt.yticks([])
    
    ani = animation.ArtistAnimation(
        fig, frames,
        interval=repeat_delay/fps, # interval in milliseconds  
        blit=True,
        repeat_delay=repeat_delay
    )
    plt.show()

def onehot2coloredmask(onehot):
    mask = torch.argmax(onehot, dim=0).numpy()
    mask = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    return mask
