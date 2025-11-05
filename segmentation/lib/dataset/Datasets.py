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
