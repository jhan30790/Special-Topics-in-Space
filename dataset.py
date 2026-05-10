"""
dataset.py
讀取 DeepMoon 的 HDF5 資料，回傳 (image, mask) 給 PyTorch DataLoader

實際 HDF5 結構：
  train_images.hdf5:
    input_images: (30000, 256, 256) uint8   ← 圖像
    target_masks: (30000, 256, 256) float32 ← mask（已預先生成）
  train_craters.hdf5: 不需要用到（mask 已內建）
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class CraterDataset(Dataset):
    def __init__(self, images_path, n_samples=None, augment=False):
        self.images_path = images_path
        self.augment = augment

        with h5py.File(images_path, "r") as f:
            total = f["input_images"].shape[0]

        self.n = total if n_samples is None else min(n_samples, total)
        print(f"[Dataset] 載入 {self.n} 張，augment={augment}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(self.images_path, "r") as f:
            img  = f["input_images"][idx]   # (256, 256) uint8
            mask = f["target_masks"][idx]   # (256, 256) float32

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()

        img_tensor  = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return img_tensor, mask_tensor


if __name__ == "__main__":
    import os

    base = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master"
    ds = CraterDataset(
        images_path=os.path.join(base, "train_images.hdf5"),
        n_samples=5,
        augment=False,
    )

    img, mask = ds[0]
    print(f"image shape : {img.shape}")
    print(f"image range : {img.min():.3f} ~ {img.max():.3f}")
    print(f"mask  shape : {mask.shape}")
    print(f"mask  max={mask.max():.3f}  min={mask.min():.3f}")
    print("dataset.py 測試通過！")
