import torch
import os
import glob
import numpy as np
import PIL.Image as Image


class DRIVE(_PaletteSegmentationMixin, torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, label_transform=None):
        'Initialization'
        if isinstance(split, bool):
            split = 'train' if split else 'test'
        if not isinstance(split, str):
            raise TypeError('split must be a bool or a string.')

        split = split.lower()
        if split == 'val':
            split = 'validate'

        valid_splits = {'train', 'validate', 'test'}
        if split not in valid_splits:
            raise ValueError(f"Unsupported split '{split}'. Choose from {sorted(valid_splits)}.")

        self.split = split
        self.transform = transform
        self.label_transform = label_transform

        base_dir = "/work3/s204164/DeepLearningCompVis/segmentation/data/DRIVE"
        split_dir = os.path.join(base_dir, split)
        image_dir = os.path.join(split_dir, 'images')
        label_dir = os.path.join(split_dir, 'labels')

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f'No DRIVE images directory found at {image_dir}.')
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f'No DRIVE labels directory found at {label_dir}.')

        image_patterns = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp')
        image_paths = []
        for pattern in image_patterns:
            image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
        self.image_paths = sorted(image_paths)
        if not self.image_paths:
            raise FileNotFoundError(f'No DRIVE images found in {image_dir}.')

        label_lookup = {}
        for pattern in ('*.png', '*.gif', '*.tif', '*.tiff'):
            for label_path in glob.glob(os.path.join(label_dir, pattern)):
                stem = os.path.splitext(os.path.basename(label_path))[0]
                parts = stem.rsplit('_', 1)
                key = parts[0] if len(parts) > 1 else stem
                label_lookup.setdefault(key, label_path)

        if not label_lookup:
            raise FileNotFoundError(f'No DRIVE labels found in {label_dir}.')

        self.samples = []
        for image_path in self.image_paths:
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            image_key_parts = image_stem.rsplit('_', 1)
            image_key = image_key_parts[0] if len(image_key_parts) > 1 else image_stem
            label_path = label_lookup.get(image_key)
            if label_path is None:
                raise FileNotFoundError(f'No matching label found for {os.path.basename(image_path)} in {label_dir}.')
            self.samples.append((image_path, label_path))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.samples)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path, label_path = self.samples[idx]

        with Image.open(image_path) as img:
            image = img.convert('RGB')
        with Image.open(label_path) as lbl:
            label = lbl.convert('L')

        if self.transform is not None:
            X = self.transform(image)
        else:
            X = self._pil_to_tensor(image)

        if self.label_transform is not None:
            Y = self.label_transform(label)
        else:
            Y = self._pil_to_tensor(label)

        return X, Y