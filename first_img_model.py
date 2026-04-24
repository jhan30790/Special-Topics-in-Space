import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ===== 路徑設定 =====
dev_images_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
model_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"

# ===== 讀模型 =====
model = tf.keras.models.load_model(model_path, compile=False)

idx = 0

with h5py.File(dev_images_path, "r") as f:
    img = f["input_images"][idx]
    mask = f["target_masks"][idx]

# ===== 前處理 =====
img_float = img.astype("float32")
img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

x = img_norm.reshape(1, 256, 256, 1)
pred_raw = model.predict(x)

if pred_raw.ndim == 4:
    pred = pred_raw[0, :, :, 0]
elif pred_raw.ndim == 3:
    pred = pred_raw[0, :, :]
else:
    raise ValueError(f"模型輸出維度異常：{pred_raw.shape}")

# ===== 二值化 =====
gt = mask > 0
threshold = 0.25 * pred.max()
pred_bin = pred > threshold

# ===== 計算 IoU / Dice =====
intersection = np.logical_and(gt, pred_bin).sum()
union = np.logical_or(gt, pred_bin).sum()

iou = intersection / (union + 1e-8)
dice = 2 * intersection / (gt.sum() + pred_bin.sum() + 1e-8)

from skimage.morphology import dilation, disk

# ===== 寬鬆版 IoU / Dice：允許邊緣誤差 =====
tol = 2  # 允許 2 pixels 誤差，可以試 2, 3, 5
se = disk(tol)

gt_dilated = dilation(gt, se)
pred_dilated = dilation(pred_bin, se)

intersection_tol = np.logical_and(gt_dilated, pred_dilated).sum()
union_tol = np.logical_or(gt_dilated, pred_dilated).sum()

iou_tol = intersection_tol / (union_tol + 1e-8)
dice_tol = 2 * intersection_tol / (gt_dilated.sum() + pred_dilated.sum() + 1e-8)

print(f"Tolerance = {tol} pixels")
print("Relaxed IoU =", iou_tol)
print("Relaxed Dice =", dice_tol)

print("threshold =", threshold)
print("GT pixels =", gt.sum())
print("Pred pixels =", pred_bin.sum())
print("Intersection =", intersection)
print("Union =", union)
print("IoU =", iou)
print("Dice =", dice)

# ===== 畫圖 =====
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].set_title("input image")
ax[0, 0].axis("off")

ax[0, 1].imshow(gt, cmap="gray")
ax[0, 1].set_title("ground truth mask")
ax[0, 1].axis("off")

ax[0, 2].imshow(pred, cmap="hot")
ax[0, 2].set_title("model heatmap")
ax[0, 2].axis("off")

ax[1, 0].imshow(pred_bin, cmap="gray")
ax[1, 0].set_title("prediction binary mask")
ax[1, 0].axis("off")

ax[1, 1].imshow(img, cmap="gray")
ax[1, 1].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
ax[1, 1].contour(pred_bin, levels=[0.5], colors="red", linewidths=1.0)
ax[1, 1].set_title("GT(green) vs Pred(red)")
ax[1, 1].axis("off")

# 重疊區域可視化
overlay = np.zeros((256, 256, 3), dtype=np.float32)
overlay[..., 1] = gt.astype(np.float32)        # green = GT
overlay[..., 0] = pred_bin.astype(np.float32)  # red = Pred

ax[1, 2].imshow(overlay)
ax[1, 2].set_title(f"Overlap\nIoU={iou:.4f}, Dice={dice:.4f}")
ax[1, 2].axis("off")

overlay_tol = np.zeros((256, 256, 3), dtype=np.float32)
overlay_tol[..., 1] = gt_dilated.astype(np.float32)        # green = GT expanded
overlay_tol[..., 0] = pred_dilated.astype(np.float32)      # red = Pred expanded

plt.figure(figsize=(6, 6))
plt.imshow(overlay_tol)
plt.title(f"Relaxed Overlap, tol={tol}px\nIoU={iou_tol:.4f}, Dice={dice_tol:.4f}")
plt.axis("off")
plt.show()

plt.tight_layout()
plt.show()
