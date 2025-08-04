import cv2
from glob import glob
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as BaseDataset
import numpy as np

class TestDataset(BaseDataset):
    def __init__(self, dir_list, transform=None, eval=True, batch_size=4, cache_size=100, train=False, num_classes=10):
        self.dir_list = dir_list
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        self.true_mask_paths = []
        self.video_starts = []
        self.eval = eval
        self.train = train
        # Add cache for images and masks
        self.cache_size = cache_size
        self.image_cache = {}
        self.mask_cache = {}
        
        # Pre-compute one-hot encoding matrix
        self.one_hot_matrix = np.eye(num_classes)
        self.num_classes = num_classes
        
        # Create a dictionary mapping video_dir to number of images
        len_dict = {}
        for video_dir in self.dir_list:
            image_paths = sorted(glob(f"{video_dir}/rgb/*.png"))
            len_dict[video_dir] = len(image_paths)
        
        # Get partitioned indices using find_partition
        if batch_size > 1:
            partitions, length = find_partition(len_dict, batch_size=batch_size)

            # imageのpathをリスト化。batch_size列の内包リストを作る
            for i, partition in enumerate(partitions):
                _image_paths = []
                for video_dir in partition:
                    image_paths = sorted(glob(f"{video_dir}/rgb/*.png"))
                    _image_paths.extend(image_paths)
                
                for i in range(length - len(_image_paths)):
                    _image_paths.append(None)

                self.image_paths.append(_image_paths)
        
            # transpose - ex. ([1,1,2,2],[3,3,3,4],[5,5,6,6]) -> (1,3,5,1,3,5,2,3,6,2,4,6)
            h, w = len(self.image_paths), len(self.image_paths[0])
            self.image_paths = [self.image_paths[i][j] for j in range(w) for i in range(h)]
            print(self.image_paths[:8])
        else:
            self.image_paths = [sorted(glob(f"{video_dir}/rgb/*.png")) for video_dir in self.dir_list]
            self.image_paths = sum(self.image_paths, [])
        
        
        if eval:
            for p in self.image_paths:
                if p is None:
                    self.mask_paths.append(None)
                    self.true_mask_paths.append(False)
                    continue
                _path = p.replace("rgb", "segmentation")
                if Path(_path).exists():
                    self.mask_paths.append(_path)
                    self.true_mask_paths.append(True)
                else:
                    self.mask_paths.append(self.mask_paths[-1])
                    self.true_mask_paths.append(False)
        # image_paths内の、ファイル名が000000000.pngのindexをTrueとするvideo_startsを作成
        if train:
            frame_counter = 0
            i = 0
            while i < len(self.image_paths):
                p = self.image_paths[i]
                frame_counter += 1
                if p is None:
                    self.mask_paths.append(None)
                    self.true_mask_paths.append(False)
                    i += 1
                    continue
                _path = p.replace("rgb", "segmentation")
                if Path(_path).exists():
                    self.mask_paths.append(_path)
                    self.true_mask_paths.append(True)
                    frame_counter = 0
                    i += 1
                else:
                    # For frames without masks, only include every 10th frame
                    if frame_counter % 10 == 0:
                        self.mask_paths.append(self.mask_paths[-1])
                        self.true_mask_paths.append(False)
                        i += 1
                    else:
                        self.image_paths.pop(i)
        
        for p in self.image_paths:
            if p is None:
                self.video_starts.append(False)
            else:
                self.video_starts.append(p.endswith("000000000.png"))
        #print(self.image_paths[206:208])
        print(len(self.image_paths), len(self.mask_paths), len(self.true_mask_paths), len(self.video_starts))
        #assert len(self.image_paths) == len(self.mask_paths) == len(self.true_mask_paths) == len(self.video_starts)

    def _load_and_process_image(self, path):
        if path is None:
            return np.zeros((1024, 1024, 3))
        
        if path in self.image_cache:
            return self.image_cache[path]
        
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024))
        
        # Cache management
        if len(self.image_cache) >= self.cache_size:
            self.image_cache.pop(next(iter(self.image_cache)))
        self.image_cache[path] = image
        
        return image

    def _load_and_process_mask(self, path):
        if path is None:
            return np.zeros((10, 1024, 1024))
        
        if path in self.mask_cache:
            return self.mask_cache[path]
        
        mask = cv2.imread(path)[:, :, 0]
        if self.num_classes == 2:
            mask = (mask > 0).astype(np.uint8)
        assert mask.max() < self.num_classes, f"Invalid mask value: {mask.max()} at {path}"
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Pre-compute one-hot encoded mask
        mask = self.one_hot_matrix[mask].transpose(2, 0, 1)
        
        # Cache management
        if len(self.mask_cache) >= self.cache_size:
            self.mask_cache.pop(next(iter(self.mask_cache)))
        self.mask_cache[path] = mask
        
        return mask

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.eval or self.train:
            # Load image and mask using cache
            image = self._load_and_process_image(self.image_paths[idx])
            mask = self._load_and_process_mask(self.mask_paths[idx])
            video_start = self.video_starts[idx]
            flag = self.true_mask_paths[idx]
            
            # Apply transforms if any
            if self.transform:
                augmented = self.transform(image=image, mask=mask.transpose(1, 2, 0))
                image = augmented['image']
                mask = augmented['mask'].transpose(2, 0, 1)
                
            # Convert to torch tensors
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)
            return image, mask, video_start, flag
        else:
            image = self._load_and_process_image(self.image_paths[idx])
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            video_start = self.video_starts[idx]
            return image, video_start
        

def find_partition(len_dict, batch_size=4):
    """
    lens: {video_id: length}
    batch_size: int - batch size
    
    Returns:
    - List[List[int]]: list of indices for each batch
    """

    sorted_lens = sorted(len_dict.items(), key=lambda x: x[1], reverse=True)
    columns = [[] for _ in range(batch_size)]
    column_sums = [0] * batch_size

    for tape in sorted_lens:
        idx = np.argmin(column_sums)
        columns[idx].append(tape[0])
        column_sums[idx] += tape[1]

    return columns, np.max(column_sums)
