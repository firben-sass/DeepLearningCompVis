import torch
import os
import glob
import numpy as np
import PIL.Image as Image


class _PaletteSegmentationMixin:
    def _pil_to_tensor(self, image):
        array = np.array(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, :, :]
        else:
            array = array.transpose(2, 0, 1)
        return torch.from_numpy(array / 255.0)

    def _palette_to_one_hot(self, label_image):
        if torch.is_tensor(label_image):
            label_np = label_image.detach().cpu().numpy()
        else:
            label_np = np.array(label_image, dtype=np.int64)

        if label_np.ndim == 3:
            label_np = label_np.squeeze()

        if label_np.dtype != np.int64:
            if label_np.max() <= 1.0:
                label_np = np.rint(label_np * 255.0)
            else:
                label_np = np.rint(label_np)
            label_np = label_np.astype(np.int64)

        label_tensor = torch.from_numpy(label_np.astype(np.int64))
        if label_tensor.dim() != 2:
            raise ValueError('Expected label tensor with 2 dimensions (H, W) after conversion.')

        if label_tensor.max() >= self.num_classes:
            raise ValueError(f'Found label id {int(label_tensor.max())} >= num_classes ({self.num_classes}).')

        one_hot = torch.zeros((self.num_classes, label_tensor.shape[0], label_tensor.shape[1]), dtype=torch.float32)
        one_hot.scatter_(0, label_tensor.unsqueeze(0), 1.0)
        return one_hot


class PhC:
    def __init__(self, train, transform):
        'Initialization'
        self.transform = transform
        root_path = '/dtu/datasets1/02516/phc_data'
        data_path = os.path.join(root_path, 'train' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))
        self.label_paths = sorted(glob.glob(data_path + '/labels/*.png'))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y


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

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'DRIVE'))
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


class PH2(_PaletteSegmentationMixin, torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, label_transform=None):
        'Initialization'
        if isinstance(split, bool):
            split = 'train' if split else 'test'
        if not isinstance(split, str):
            raise TypeError('split must be a bool or a string.')

        split = split.lower()
        if split == 'val':
            split = 'validate'

        valid_splits = ('train', 'validate', 'test')
        if split not in valid_splits:
            raise ValueError(f"Unsupported split '{split}'. Choose from {valid_splits}.")

        self.split = split
        self.transform = transform
        self.label_transform = label_transform

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'PH2_Dataset_images'))
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f'PH2 dataset directory not found at {base_dir}.')

        case_dirs = sorted([path for path in glob.glob(os.path.join(base_dir, 'IMD*')) if os.path.isdir(path)])
        if not case_dirs:
            raise RuntimeError(f'No PH2 cases found under {base_dir}.')

        # Gather sample metadata before splitting so we can determine a consistent crop.
        all_samples = []
        for case_dir in case_dirs:
            image_candidates = sorted(glob.glob(os.path.join(case_dir, '*_Dermoscopic_Image', '*.bmp')))
            if not image_candidates:
                raise FileNotFoundError(f'No dermoscopic image found in {case_dir}.')
            label_candidates = sorted(glob.glob(os.path.join(case_dir, '*_lesion', '*_lesion.bmp')))
            if not label_candidates:
                raise FileNotFoundError(f'No lesion mask found in {case_dir}.')

            image_path = image_candidates[0]
            label_path = label_candidates[0]
            with Image.open(image_path) as img:
                width, height = img.size
            all_samples.append({
                'case_dir': case_dir,
                'image_path': image_path,
                'label_path': label_path,
                'width': width,
                'height': height,
            })

        if not all_samples:
            raise RuntimeError(f'No PH2 samples were collected from {base_dir}.')

        self.target_width = min(sample['width'] for sample in all_samples)
        self.target_height = min(sample['height'] for sample in all_samples)

        split_names = ('train', 'validate', 'test')
        split_arrays = np.array_split(np.array(case_dirs, dtype=object), len(split_names))
        split_map = {
            name: list(arr.tolist()) if len(arr) else []
            for name, arr in zip(split_names, split_arrays)
        }

        self.case_dirs = split_map.get(self.split, case_dirs)
        if not self.case_dirs:
            self.case_dirs = case_dirs

        selected_case_dirs = set(self.case_dirs)
        self.samples = []
        for sample in all_samples:
            if sample['case_dir'] not in selected_case_dirs:
                continue
            crop_box = self._center_crop_box(sample['width'], sample['height'])
            self.samples.append((sample['image_path'], sample['label_path'], crop_box))

        if not self.samples:
            raise RuntimeError(f'No PH2 samples were collected from {base_dir}.')

    def __len__(self):
        'Returns the total number of samples'
        return len(self.samples)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path, label_path, crop_box = self.samples[idx]

        with Image.open(image_path) as img:
            image = img.convert('RGB')
        with Image.open(label_path) as lbl:
            label = lbl.convert('L')

        if crop_box is not None:
            image = image.crop(crop_box)
            label = label.crop(crop_box)

        if self.transform is not None:
            X = self.transform(image)
        else:
            X = self._pil_to_tensor(image)

        if self.label_transform is not None:
            Y = self.label_transform(label)
        else:
            Y = self._pil_to_tensor(label)

        return X, Y

    def _center_crop_box(self, width, height):
        if width == self.target_width and height == self.target_height:
            return None
        left = max(0, (width - self.target_width) // 2)
        top = max(0, (height - self.target_height) // 2)
        right = left + self.target_width
        bottom = top + self.target_height
        return (left, top, right, bottom)


class CMP(_PaletteSegmentationMixin, torch.utils.data.Dataset):
    def __init__(self, train, transform=None, label_transform=None, num_classes=12):
        'Initialization'
        self.transform = transform
        self.label_transform = label_transform
        self.num_classes = num_classes
        root_path = '/dtu/datasets1/02516/CMP_facade_DB_base/base'
        self.image_paths = sorted(glob.glob(root_path + '/cmp_b*.jpg'))
        self.label_paths = sorted(glob.glob(root_path + '/cmp_b*.png'))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform is not None:
            X = self.transform(image)
        else:
            X = self._pil_to_tensor(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        Y = self._palette_to_one_hot(label)
        return X, Y


