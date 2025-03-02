import os
import torch
import numpy as np
from torch.utils.data import Dataset

class GridData(Dataset):
    def __init__(self, path, clip_value=0.95, img_size=64):
        self.img_size = img_size
        self.clip_v = clip_value

        self.maps = np.load(os.path.join(path, 'maps.npy'), mmap_mode='c')
        self.goals = np.load(os.path.join(path, 'goals.npy'), mmap_mode='c')
        self.starts = np.load(os.path.join(path, 'starts.npy'), mmap_mode='c')
        self.gt_values = np.load(os.path.join(path, 'focal.npy'), mmap_mode='c')

    def __len__(self):
        return len(self.gt_values)

    def __getitem__(self, idx):
        gt_ = torch.from_numpy(self.gt_values[idx].astype('float32'))
        if self.clip_v:
            gt_ = torch.where(gt_ >= self.clip_v, gt_, torch.zeros_like(gt_))
            
        return (
            torch.from_numpy(self.maps[idx].astype('float32')), 
            torch.from_numpy(self.starts[idx].astype('float32')), 
            torch.from_numpy(self.goals[idx].astype('float32')), 
            gt_
        )
