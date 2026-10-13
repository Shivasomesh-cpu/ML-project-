import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib

class CAT12VBMDataset(Dataset):
    """
    Dataset for modulated warped grey matter maps (mwp1*.nii) from CAT12 VBM.
    Target spatial resolution: (121, 128, 121).
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.target_shape = (121, 128, 121)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = nib.load(self.data[idx]).get_fdata().astype(np.float32)
        if img.shape != self.target_shape:
            img = F.interpolate(
                torch.from_numpy(img[None, None]),
                size=self.target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze().numpy()

        if self.transform is not None:
            img = self.transform(img)
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            if img.ndim == 3:
                img = img.unsqueeze(0)
        else:
            img = torch.from_numpy(img).float()
            if img.ndim == 3:
                img = img.unsqueeze(0)
        return img


class BrainIACDataset(Dataset):
    """
    Dataset for BrainIAC ViT.
    Target spatial resolution: (96, 96, 96).
    """
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self.target_shape = (96, 96, 96)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        if img.shape != self.target_shape:
            img = F.interpolate(
                torch.from_numpy(img[None, None]),
                size=self.target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze().numpy()

        img_tensor = torch.from_numpy(img).float()
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        else:
            img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor


class SwinBrainDataset(Dataset):
    """
    Dataset for SwinBrain (SwinUNETR).
    Target spatial resolution: (128, 128, 64) with 3 duplicated channels.
    """
    def __init__(self, paths):
        self.paths = paths
        self.target_shape = (128, 128, 64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        img = F.interpolate(
            torch.from_numpy(img[None, None]),
            size=self.target_shape,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Shape: (1, 128, 128, 64)
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = img.repeat(3, 1, 1, 1)  # Shape: (3, 128, 128, 64)
        return img


class CNNDataset(Dataset):
    """
    Dataset for custom 3D CNN baseline.
    Target spatial resolution: (91, 109, 91).
    """
    def __init__(self, paths):
        self.paths = paths
        self.target_shape = (91, 109, 91)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        if img.shape != self.target_shape:
            img = F.interpolate(
                torch.from_numpy(img[None, None]),
                size=self.target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze().numpy()
        img = (img - img.mean()) / (img.std() + 1e-6)
        return torch.from_numpy(img[None]).float()


class TurboPrepDataset(Dataset):
    """
    Dataset for 3D-Neuro-SimCLR ResNet backbone.
    Target resolution: center-crop (150, 192, 192).
    """
    def __init__(self, paths):
        self.paths = paths
        self.target_shape = (150, 192, 192)

    def __len__(self):
        return len(self.paths)

    def _center_crop_or_pad(self, img, target_shape):
        # Transpose or crop to match target
        result = np.zeros(target_shape, dtype=img.dtype)
        slices_in = []
        slices_out = []
        for s_curr, s_tgt in zip(img.shape, target_shape):
            if s_curr >= s_tgt:
                start = (s_curr - s_tgt) // 2
                slices_in.append(slice(start, start + s_tgt))
                slices_out.append(slice(0, s_tgt))
            else:
                start = (s_tgt - s_curr) // 2
                slices_in.append(slice(0, s_curr))
                slices_out.append(slice(start, start + s_curr))
        result[tuple(slices_out)] = img[tuple(slices_in)]
        return result

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        img = self._center_crop_or_pad(img, self.target_shape)
        img = (img - img.mean()) / (img.std() + 1e-6)
        return torch.from_numpy(img[None]).float()
