import numpy as np
import pandas as pd 
from PIL import Image
import os
import tqdm
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
#from .view_generator import ContrastiveLearningViewGenerator
from .gaussian_blur import GaussianBlur
from torchvision.transforms import functional as F
import torch.multiprocessing
import warnings
import json

torch.multiprocessing.set_sharing_strategy('file_system')


class ContrastStretch(torch.nn.Module):
    def __init__(self, low_percentile=0.05, high_percentile=0.95):
        super().__init__()
        self.low = low_percentile
        self.high = high_percentile

    def forward(self, x):
        # x: Tensor [C, H, W]
        x_reshaped = x.view(x.size(0), -1)  # [C, H*W]
        low_vals = torch.quantile(x_reshaped, self.low, dim=1, keepdim=True)
        high_vals = torch.quantile(x_reshaped, self.high, dim=1, keepdim=True)
        x_stretched = (x_reshaped - low_vals) / (high_vals - low_vals + 1e-6)
        x_stretched = x_stretched.clamp(0.0, 1.0)
        return x_stretched.view_as(x)

class LogTransform(torch.nn.Module):
    def forward(self, x):
        return torch.log1p(x)

class ZScoreClipping(torch.nn.Module):
    def __init__(self, min_val=-3.0, max_val=3.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clip(x, self.min_val, self.max_val)

class MultiChannelColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.1, hue=0.5):
        self.color_transform = v2.ColorJitter(brightness=brightness, 
                                              contrast=contrast, 
                                              saturation=saturation,
                                              hue=hue)
        
    def __call__(self, image):
        enhanced_channels = [self.color_transform(c) for c in image.split()]
        return Image.merge(image.mode, enhanced_channels)

class MultiModalCancerDataset(Dataset):
    def __init__(self, root_path, df, mode='BF', split='train', size=256, used_channels=None, not_masked_channels=None, partition_name=None, norm_config_path=None, input_norm="zscore", BF_aug=None, FL_aug=None):
        self.root = root_path
        self.df = df
        self.mode = mode  # 'BF', 'FL', 'MM'
        self.split = split
        self.size = size
        self.input_norm = input_norm.lower()
        self.used_channels = [int(ch) for ch in used_channels] if used_channels else None
        self.not_masked_channels = [int(ch) for ch in not_masked_channels] if not_masked_channels else None
        #
        if self.split == "train":
            self.BF_aug = BF_aug if BF_aug is not None else ["posterize", "gaussian", "solarize", "sharpness", "autocontrast", "erase"]
            self.FL_aug = FL_aug if FL_aug is not None else ["colorjitter", "erase"]
        else:
            self.BF_aug = []
            self.FL_aug = []

        print(f"[INIT] Split: {self.split}, BF_aug: {self.BF_aug}, FL_aug: {self.FL_aug}")

        # Normalization values (channel-wise mean/std)
        self.normalize_BF = None
        self.normalize_FL = None

        #NEW NORM
        # Load normalization parameters from JSON
        if partition_name and norm_config_path:
            with open(norm_config_path, 'r') as f:
                norms = json.load(f)
                part = norms.get(partition_name)
                if part:
                    if 'BF' in part:
                        self.normalize_BF = v2.Normalize(mean=part['BF']['mean'], std=part['BF']['std'])
                    if 'FL' in part:
                        self.normalize_FL = v2.Normalize(mean=part['FL']['mean'], std=part['FL']['std'])
                else:
                    raise ValueError(f"Partition '{partition_name}' not found in {norm_config_path}")
                
        # Fallback default values if not set - wenyis values
        if self.normalize_BF is None:
            self.normalize_BF = v2.Normalize((0.5251, 0.5998, 0.6397), (0.2339, 0.1905, 0.1573))
        if self.normalize_FL is None:
            self.normalize_FL = v2.Normalize((0.0804, 0.0489, 0.1264, 0.1098), (0.0732, 0.0579, 0.0822, 0.0811))

        # Build transforms
        self.transform_train_common = self.get_common_transform() if split == 'train' else None
        self.transform_BF = self.get_bf_transform(train=(split == 'train'))
        self.transform_FL = self.get_fl_transform(train=(split == 'train'))

        print(f"self.split = {self.split}")
        print(f"self.transform_train = {self.transform_train_common}")
        print(f"self.transform_BF = {self.transform_BF}")
        print(f"self.transform_FL = {self.transform_FL}")
        print(f'self.mode = {self.mode}')
    
    def get_normalization_transform(self, norm_type, is_bf=True):
        norm_type = norm_type.lower()
        transforms_list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]

        if norm_type == "scale01":
            return v2.Compose(transforms_list)
         
        elif norm_type == "zscore":
            transforms_list.append(self.normalize_BF if is_bf else self.normalize_FL)

        elif norm_type == "zscore_clipped":
            transforms_list.append(self.normalize_BF if is_bf else self.normalize_FL)
            transforms_list.append(ZScoreClipping(-3, 3))
        
        elif norm_type == "log_zscore_clipped":
            transforms_list.append(LogTransform())
            transforms_list.append(self.normalize_BF if is_bf else self.normalize_FL)
            transforms_list.append(ZScoreClipping(-3, 3))

        elif norm_type == "contrast_stretch":
            transforms_list.append(ContrastStretch(0.05, 0.95))
      
        else:
            raise ValueError(f"Unknown input_norm type: {norm_type}")

        return v2.Compose(transforms_list)
        
    def get_common_transform(self):
        return v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.CenterCrop(size=self.size),
            #v2.RandomResizedCrop(size=self.size, scale=(1.0, 1.0), antialias=True),
            #v2.RandomCrop(size=self.size),
        ])

    def get_bf_transform(self, train=True):
        aug = []
        if train:
            if any(op in self.BF_aug for op in ["posterize", "solarize", "gaussian"]):
                choices = []
                if "posterize" in self.BF_aug:
                    choices.append(v2.RandomPosterize(3))
                if "gaussian" in self.BF_aug:
                    choices.append(v2.GaussianBlur(5, 1.5))
                if "solarize" in self.BF_aug:
                    choices.append(v2.RandomSolarize(100))
                if choices:
                    aug.append(v2.RandomChoice(choices, p=[1 / len(choices)] * len(choices)))

            if "colorjitter" in self.BF_aug:
                aug.append(v2.ColorJitter(0.5, 0.2, 0.2, 0.2))

            if "sharpness" in self.BF_aug:
                aug.append(v2.RandomAdjustSharpness(sharpness_factor=2))

            if "autocontrast" in self.BF_aug:
                aug.append(v2.RandomAutocontrast())

            if "erase" in self.BF_aug:
                aug.append(v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'))

        aug.append(self.get_normalization_transform(self.input_norm, is_bf=True))
        if not train:
            aug.append(v2.CenterCrop(size=self.size))
        return v2.Compose(aug)

    def get_fl_transform(self, train=True):
        aug = []
        if train:
            if "colorjitter" in self.FL_aug:
                aug.append(MultiChannelColorJitter(0.8, 0.8, 0.8, 0.5))
            if "gaussian" in self.FL_aug:
                aug.append(v2.GaussianBlur(5, sigma=(0.3, 3.2)))
            if "erase" in self.FL_aug:
                aug.append(v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'))

        aug.append(self.get_normalization_transform(self.input_norm, is_bf=False))
        if not train:
            aug.append(v2.CenterCrop(size=self.size))
        return v2.Compose(aug)

    
    def __len__(self):
        return len(self.df)

    def get_ratio(self):
        ratio = self.df['Diagnosis'].value_counts(normalize=True)
        return ratio.get(1, 0.0), ratio.get(0, 0.0)

    def load_image(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return Image.open(path)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        slide = data['Slide']
        patch = data['Patch']
        label = data['Diagnosis']
        name = f"{slide}_{patch}"

        BF_img = FL_img = MM_img = None

        # Load and transform images
        if self.mode in ['BF', 'MM']:
            bf_path = os.path.join(self.root, f'BF/{slide:02d}/patch_{patch}.tif')
            BF_img = self.transform_BF(self.load_image(bf_path))

        if self.mode in ['FL', 'MM']:
            fl_path = os.path.join(self.root, f'FL/{slide:02d}/patch_{patch}.tif')
            FL_img = self.transform_FL(self.load_image(fl_path))

        if self.mode == 'MM':
            MM_img = torch.cat([BF_img, FL_img], dim=0)

        # Apply shared transform (e.g., crop/flip)
        if self.split == 'train' and self.transform_train_common:
            if self.mode == 'BF':
                BF_img = self.transform_train_common(BF_img)
            elif self.mode == 'FL':
                FL_img = self.transform_train_common(FL_img)
            elif self.mode == 'MM':
                MM_img = self.transform_train_common(MM_img)

        # Select final image
        final_img = {
            'BF': BF_img,
            'FL': FL_img,
            'MM': MM_img
        }.get(self.mode)

        if self.mode == 'MM' and self.used_channels is not None and final_img is not None:
            final_img = final_img[self.used_channels]
        
        # Channel occlusion (optional)
        #if self.mode == 'MM' and self.not_masked_channels is not None and final_img is not None:
        if self.not_masked_channels is not None and final_img is not None:
            masked_img = final_img.clone()
            active_channels = self.not_masked_channels  # list of indices to keep
            all_channels = torch.arange(masked_img.shape[0])

            # Channels not in used_channels → set to zero
            for ch in all_channels:
                if ch.item() not in active_channels:
                    masked_img[ch] = 0.0  # Occlusion: overwrite with zero

            final_img = masked_img


        return {'image': final_img, 'label': label, 'slide': slide, 'name': name}