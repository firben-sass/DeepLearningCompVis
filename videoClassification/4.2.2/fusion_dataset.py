import os
from glob import glob
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
import random

class RGBFlowPairDataset(Dataset):
    """
    Pairs a single RGB frame with its corresponding optical flow stack from the same video.
    Assumes structure under root_dir:
      - frames/{split}/{class}/{video_name}/frame_*.jpg
      - flows/{split}/{class}/{video_name}/flow_*.npy   # each is shape (2, H, W)
    """
    def __init__(self,
                 root_dir: str,
                 split: str = "train",
                 image_size: int = 224,
                 n_frames: int = 10,
                 random_frame: bool = True,
                 aug_rgb: bool = True,
                 aug_flow: bool = True,
                 class_to_idx: Optional[dict] = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.n_frames = n_frames
        self.random_frame = random_frame
        self.aug_rgb = aug_rgb
        self.aug_flow = aug_flow

        # discover all videos from flow side (as ground truth list)
        self.flow_root = os.path.join(root_dir, "flows", split)
        self.frame_root = os.path.join(root_dir, "frames", split)

        self.video_dirs = sorted(glob(os.path.join(self.flow_root, "*", "*")))
        if len(self.video_dirs) == 0:
            raise FileNotFoundError(f"No videos found under {self.flow_root}")

        # build class mapping
        if class_to_idx is None:
            classes = sorted(list({vp.split(os.sep)[-2] for vp in self.video_dirs}))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        # rgb transforms
        if split == "train" and self.aug_rgb:
            self.rgb_tf = T.Compose([
                T.Resize(int(image_size*1.1)),
                T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.rgb_tf = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.hflip_prob = 0.5 if (split == "train" and self.aug_flow) else 0.0

    def __len__(self) -> int:
        return len(self.video_dirs)

    def _load_one_frame(self, frames_dir: str) -> Image.Image:
        frames = sorted(glob(os.path.join(frames_dir, "frame_*.jpg")))
        if len(frames) == 0:
            raise FileNotFoundError(f"No frames in {frames_dir}")
        if self.random_frame:
            frame_path = random.choice(frames)
        else:
            mid = len(frames)//2
            frame_path = frames[mid]
        return Image.open(frame_path).convert("RGB")

    def _load_flows(self, flow_dir: str) -> torch.Tensor:
        flow_files = sorted(glob(os.path.join(flow_dir, "flow_*.npy")))
        if len(flow_files) == 0:
            raise FileNotFoundError(f"No flow frames found in {flow_dir}")

        # optionally sample up to n_frames-1 flow pairs (each file is 2 channels)
        if self.n_frames is not None and len(flow_files) > (self.n_frames - 1):
            # uniform sample indices
            idxs = np.linspace(0, len(flow_files) - 1, self.n_frames - 1).astype(int).tolist()
            flow_files = [flow_files[i] for i in idxs]

        flows = []
        for fp in flow_files:
            f = np.load(fp)   # (2, H, W) -> (u, v)
            f = np.clip(f, -20, 20) / 20.0
            f = torch.from_numpy(f).float()  # [2, H, W]
            f = F.interpolate(f.unsqueeze(0), size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False).squeeze(0)
            flows.append(f)
        # [2, H, W] * (T-1) -> [2*(T-1), H, W]
        return torch.cat(flows, dim=0)

    def _maybe_hflip_flow(self, flow: torch.Tensor, do_flip: bool) -> torch.Tensor:
        if not do_flip:
            return flow
        # flow shape: [2*(T-1), H, W]; even idx -> u (x), odd -> v (y)
        flow = torch.flip(flow, dims=[-1])  # horizontal flip
        # invert x component
        u_channels = torch.arange(0, flow.size(0), 2)
        flow[u_channels] = -flow[u_channels]
        return flow

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        flow_dir = self.video_dirs[idx]
        cls_name = flow_dir.split(os.sep)[-2]
        vid_name = flow_dir.split(os.sep)[-1]
        label = self.class_to_idx[cls_name]

        # RGB
        frames_dir = os.path.join(self.frame_root, cls_name, vid_name)
        rgb_img = self._load_one_frame(frames_dir)
        rgb_tensor = self.rgb_tf(rgb_img)

        # FLOW
        flow_tensor = self._load_flows(flow_dir)
        # stochastic hflip applied to both rgb and flow in sync is tricky because rgb_tf already flipped randomly.
        # To keep streams roughly aligned, we only apply explicit flow flip, but not required to match rgb exactly for classification.
        do_hflip_flow = (random.random() < self.hflip_prob)
        flow_tensor = self._maybe_hflip_flow(flow_tensor, do_hflip_flow)

        return rgb_tensor, flow_tensor, label
