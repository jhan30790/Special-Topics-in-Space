"""
dataset.py
讀取 DeepMoon 的 HDF5 資料，回傳 (image, mask) 給 PyTorch DataLoader
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CraterDataset(Dataset):
    """
    Args:
        images_path: train_images.hdf5 / dev_images.hdf5 / test_images.hdf5
        craters_path: train_craters.hdf5 / dev_craters.hdf5 / test_craters.hdf5
        n_samples: 只取前 n 張（筆電測試用，None = 全部）
        augment: 是否做資料增強（只在訓練集用）
    """

    def __init__(self, images_path, craters_path, n_samples=None, augment=False):
        self.images_path = images_path
        self.craters_path = craters_path
        self.augment = augment

        # 取得所有 key（img_00000, img_00001, ...）
        with h5py.File(images_path, "r") as f:
            all_keys = sorted(f.keys())

        if n_samples is not None:
            all_keys = all_keys[:n_samples]

        self.keys = all_keys
        self.n = len(self.keys)
        print(f"[Dataset] 載入 {self.n} 張圖，augment={augment}")

    def __len__(self):
        return self.n

    def _make_mask(self, craters, img_size=256):
        """
        把 crater 標註轉成 binary mask
        craters: numpy array，shape (N, 6)，欄位 [Diameter(km), Lat, Long, x, y, Diameter(pix)]
        畫圓形 mask，半徑 = Diameter(pix) / 2
        """
        mask = np.zeros((img_size, img_size), dtype=np.float32)

        if craters is None:
            return mask

        for row in craters:
            x, y, diam_pix = row[3], row[4], row[5]
            r = diam_pix / 2.0

            # 過濾無效值（無 crater 的圖會是 5e-324）
            if r < 1 or x < 0 or y < 0:
                continue

            # 畫實心圓
            cx, cy = int(round(x)), int(round(y))
            ri = int(round(r))

            # 建立 grid 計算距離
            y_grid, x_grid = np.ogrid[:img_size, :img_size]
            dist = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
            mask[dist <= r ** 2] = 1.0

        return mask

    def _is_valid_craters(self, arr):
        """
        檢查 crater 資料是否有效
        無效情況：shape (1,1) 且值接近 5e-324
        """
        if arr.shape == (1, 1):
            return False
        if arr.shape[0] == 1 and np.all(np.abs(arr) < 1e-300):
            return False
        return True

    def __getitem__(self, idx):
        key = self.keys[idx]

        # 讀圖像
        with h5py.File(self.images_path, "r") as f:
            img = f[key]["input_images"][0]  # shape: (256, 256)，灰階

        # 讀 crater 標註
        with h5py.File(self.craters_path, "r") as f:
            crater_data = f[key]["block0_values"][:]

        # 轉換圖像：正規化到 0~1
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # 建立 mask
        if self._is_valid_craters(crater_data):
            mask = self._make_mask(crater_data)
        else:
            mask = np.zeros((256, 256), dtype=np.float32)

        # 資料增強（水平/垂直翻轉，90度旋轉）
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        # 轉成 tensor：shape (1, 256, 256)
        img_tensor = torch.from_numpy(img).unsqueeze(0)    # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor


# ── 快速測試 ──────────────────────────────────────────────
if __name__ == "__main__":
    import os

    base = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master"
    ds = CraterDataset(
        images_path=os.path.join(base, "train_images.hdf5"),
        craters_path=os.path.join(base, "train_craters.hdf5"),
        n_samples=5,
        augment=False,
    )

    img, mask = ds[0]
    print(f"image shape : {img.shape}")   # 預期 torch.Size([1, 256, 256])
    print(f"image range : {img.min():.3f} ~ {img.max():.3f}")  # 0~1
    print(f"mask  shape : {mask.shape}")
    print(f"mask  unique: {mask.unique()}")  # tensor([0.]) 或 tensor([0., 1.])
    print("dataset.py 測試通過！")
