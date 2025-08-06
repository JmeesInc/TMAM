# TMAM

This is the official repository for Temporal Memory Augmentation Module (TMAM), which enables any encoder-decoder semantic segmentation models to address temporal information in Video Segmentation. The original paper is available at [url]
As the supplemental material, the inference movie is available [here](https://drive.google.com/file/d/1Y9qyMr7VWfpyzTdfXWfc64IiFhmwxKDA/view?usp=drive_link). Input, w/ TMAM, w/o TMAM, GT, from left to right. 
## Environmental setup

```bash
uv sync
source .venv/bin/activate
```

## How to Use TMAM?

```python
import sys
sys.path.append("./sam2")
from util.model import TMAM
from sam2.build_sam import build_sam2_video_predictor

...

predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", 
        "sam2/checkpoints/sam2.1_hiera_large.pt",
          device="cuda",)
model = TMAM(encoder, decoder, segmentation_head, predictor=preductor)
```
To reproduce our results, please donwloads SAR-RARP50 Dataset or CholecSeg8k Daraset here [https://www.kaggle.com/datasets/newslab/cholecseg8k]

## SAR-RARP50

1. `python train.py` # for image segmentation model
2. `python TMAM_train.py`
3. `python valid.py`
4. `python calc_score.py`

## CholecSeg8k

1. Split Train and Valid: `python cholec_split.py`
2. `python train_cholec.py` # for image segmentation model
3. `python TMAM_cholec.py`
4. `python valid_cholec.py`
5. `python calc_score_cholec.py`

## Temporal Consistency Analysis

```bash
cd tmp_cst
bash eval_tc_submit.sh
bash eval_tc_cholec_submit.sh
```
