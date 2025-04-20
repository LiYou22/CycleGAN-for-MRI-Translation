import os
import nibabel as nib
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms


class MRIT1T2Dataset(Dataset):
    def __init__(self, t1_dir, t2_dir, slice_mode='middle', paired=True, transform=None, cache_size=0,
                    file_list=None):
        """
        Args:
            t1_dir: Directory containing T1 scans
            t2_dir: Directory containing T2 scans
            slice_mode: 'middle' or 'random'
            paired: If True, uses paired T1-T2 data, else random unpaired
            transform: Optional transforms
            cache_size: Number of volumes to cache in memory (0 for no caching)
            split: If provided, should be either 'train' or 'val'. If None, use all data.
            split_ratio: Ratio of data to use for training (e.g., 0.8 means 80% training, 20% validation)
            random_seed: Seed for reproductivity.
        """
        super().__init__()
        assert slice_mode == 'middle' or slice_mode == 'random'

        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.transform = transform
        self.slice_mode = slice_mode
        self.paired = paired
        self.cache_size = cache_size

        self.t1_files = sorted([f for f in os.listdir(t1_dir) if f.endswith('.nii.gz')])
        self.t2_files = sorted([f for f in os.listdir(t2_dir) if f.endswith('.nii.gz')])

        if self.paired:
            self.paired_files = []
            for t1f in self.t1_files:
                subject_id = t1f.split('-')[0][3:]
                matching_t2 = [t2f for t2f in self.t2_files if t2f.split('-')[0][3:] == subject_id]
                if matching_t2:
                    self.paired_files.append((t1f, matching_t2[0]))
            print(f"Found {len(self.paired_files)} paired T1/T2 datasets")
            self.data_files = file_list if file_list is not None else self.paired_files
            print(f"Length of data files in this dataset : {len(self.data_files)}")
        else:
            self.data_files = [(t1f, None) for t1f in self.t1_files]

        self.cache = {}
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        t1_file, t2_file = self.data_files[idx]
        t1_vol, t1_stats = self._load_and_validate_volume(t1_file, is_t1=True)
        
        if self.paired:
            t2_vol, t2_stats = self._load_and_validate_volume(t2_file, is_t1=False)
        else:
            random_t2_idx = random.randint(0, len(self.t2_files) - 1)
            t2_vol, t2_stats = self._load_and_validate_volume(self.t2_files[random_t2_idx], is_t1=False)
            
        t1_slice_idx = self.get_slice_idx(t1_vol)
        t2_slice_idx = self.get_slice_idx(t2_vol) if not self.paired else min(t1_slice_idx, t2_vol.shape[2]-1)

        t1_slice = self.normalize_slice(t1_vol[:,:,t1_slice_idx], t1_stats)
        t2_slice = self.normalize_slice(t2_vol[:,:,t2_slice_idx], t2_stats)

        t1_tensor = torch.from_numpy(t1_slice).float().unsqueeze(0)
        t2_tensor = torch.from_numpy(t2_slice).float().unsqueeze(0)

        if self.transform:
            t1_tensor = self.transform(t1_tensor)
            t2_tensor = self.transform(t2_tensor)

        if t1_file not in self.cache:
            del t1_vol
        if t2_file not in self.cache:
            del t2_vol

        return {'T1': t1_tensor, 'T2': t2_tensor}

    def get_slice_idx(self, volume):
        """Get slice index based on slice mode."""
        if self.slice_mode == 'middle':
            return volume.shape[2] // 2
        else:
            return random.randint(0, volume.shape[2] - 1)

    def _load_and_validate_volume(self, filename, is_t1=True):
        """Load and validate a single volume"""
        dir_path = self.t1_dir if is_t1 else self.t2_dir
        filepath = os.path.join(dir_path, filename)
        
        if filepath in self.cache:
            vol = self.cache[filepath]
        else:
            vol = nib.load(filepath).get_fdata()
            
            if not self.is_valid_volume(vol):
                raise ValueError(f"Invalid volume: {filename}")
            
            if len(self.cache) < self.cache_size:
                self.cache[filepath] = vol
    
        stats = (float(vol.min()), float(vol.max()))
        
        return vol, stats
    
    def is_valid_volume(self, vol):
        """Check if volume meets quality criteria"""
        min_size = 64
        return (vol.shape[0] >= min_size and 
                vol.shape[1] >= min_size and 
                vol.shape[2] >= 1 and 
                not np.any(np.isnan(vol)))

    @staticmethod
    def normalize_slice(slice_data, stats):
        """Normalize slice to [0,1]."""
        min_val, max_val = stats
        return (slice_data - min_val) / (max_val - min_val + 1e-8)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
