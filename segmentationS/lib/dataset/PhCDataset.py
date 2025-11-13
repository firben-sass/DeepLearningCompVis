import torch
import os
import glob
import PIL.Image as Image

# Use relative path from the script location
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'phc_data')

class PhC(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(DATA_PATH, 'train' if train else 'test')
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
