from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        print(f"Found {len(self.frame_paths)} frames in {root_dir}/{split}")

       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101', 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

class OpticalFlowDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, n_frames=10):
        self.root_dir = os.path.join(root_dir, f"flows_png/{split}")
        self.transform = transform
        self.n_frames = n_frames

        # Get all video folders
        self.video_paths = sorted(glob(os.path.join(self.root_dir, '*', '*')))  # class_name/video_name
        self.classes = sorted(list(set([vp.split('/')[-2] for vp in self.video_paths])))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.class_to_idx[video_path.split('/')[-2]]

        # Get all flow frames in the video folder, sorted
        flow_files = sorted(glob(os.path.join(video_path, 'flow_*.png')))
        if len(flow_files) == 0:
            raise FileNotFoundError(f"No flow frames found in {video_path}")

        frames = []
        for flow_file in flow_files:
            img = Image.open(flow_file).convert('L')  # single-channel flow
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            frames.append(img)

        # Stack frames along channel dimension: [C=number of frames, H, W]
        frames = torch.cat(frames, dim=0)
        return frames, label

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/home/datawa/Repositories/ucf101_noleakage'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
            
