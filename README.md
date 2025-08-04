# TMAM

This is the official repository for Temporal Memory Augmen
tation Module (TMAM), which enables any encoder-decoder semantic segmentation models to address temporal information in Video Segmentation. The original paper is available at [url]

## Environmental setup

```bash
uv sync
source .venv/bin/activate
```

## How to Use TMAM?

```python
from util.model import TMAM

model = TMAM(encoder, decoder, segmentation_head, depth=depth)  # where depth is the number of encoder block
```

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