from glob import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

all_list = []
labels = [0, 5,11,12,13,21,22,23,24,25,31,32,33,50, 255]

for d1 in glob(f"./CholecSeg8k/**/"):
    for file in glob(f"{d1}/**/*_watershed_mask.png"):
        file_dict = {"group": d1.split("/")[-2], "file": file, "frame_id": int(file.split("/")[-1].split("_")[1]), "subgroup": file.split("/")[-2]}
        a = cv2.imread(file)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        for lbl in labels:
            file_dict[lbl] = (a == lbl).sum() > 0
        for i in np.unique(a):
            if i not in labels:
                print(f"----{file} has label {i} ")
        all_list.append(file_dict)

df = pd.DataFrame(all_list)

excluded = df[~df['group'].isin(['video01', 'video09', 'video12'])].drop(columns=[0])
video01 = df[df['group'] == 'video01'].drop(columns=[0]).sort_values(by="frame_id")
video09 = df[df['group'] == 'video09'].drop(columns=[0]).sort_values(by="frame_id")
video12 = df[df['group'] == 'video12'].drop(columns=[0]).sort_values(by="frame_id")
excluded.groupby("group").sum()
excluded['stratify'] = excluded[32] + excluded[13]*2

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(excluded, excluded['stratify'], groups=excluded['group']):
    excluded_train = excluded.iloc[train_index]
    excluded_test = excluded.iloc[test_index]
    break

included = pd.concat([video01, video09, video12]) # These videos have unique mask classes -> need to split separately
included['stratify'] = included[5] + included[23]*2 + included[24]*4 + included[25]*8 + included[33]*16

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(included, included['stratify'], groups=included['subgroup']):
    included_train = included.iloc[train_index]
    included_test = included.iloc[test_index]

train = pd.concat([included_train.drop(columns=["stratify", "subgroup"]), excluded_train.drop(columns=["stratify", "subgroup"])]).reset_index(drop=True)
test = pd.concat([included_test.drop(columns=["stratify", "subgroup"]), excluded_test.drop(columns=["stratify", "subgroup"])]).reset_index(drop=True)

train.to_csv("cholecseg8k_train.csv", index=False)
test.to_csv("cholecseg8k_test.csv", index=False)
