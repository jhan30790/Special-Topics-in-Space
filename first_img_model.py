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

# ===== 前處理（模型輸入）=====
img_float = img.astype("float32")
img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

x = img_norm.reshape(1, 256, 256, 1)
pred_raw = model.predict(x, verbose=0)

if pred_raw.ndim == 4:
    pred = pred_raw[0, :, :, 0]
elif pred_raw.ndim == 3:
    pred = pred_raw[0, :, :]
else:
    raise ValueError(f"模型輸出維度異常：{pred_raw.shape}")

# ===== 強化後影像（只拿來顯示，不拿來餵模型）=====
try:
    from skimage import exposure
    img_enhanced = exposure.equalize_adapthist(img_norm, clip_limit=0.03)
except ModuleNotFoundError:
    print("沒有安裝 scikit-image，改用簡單 contrast stretch")
    p2, p98 = np.percentile(img_norm, (2, 98))
    img_enhanced = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

# ===== 二值化 =====
gt = mask > 0
threshold = 0.25 * pred.max()
pred_bin = pred > threshold

# ===== 計算 IoU / Dice =====
intersection = np.logical_and(gt, pred_bin).sum()
union = np.logical_or(gt, pred_bin).sum()

iou = intersection / (union + 1e-8)
dice = 2 * intersection / (gt.sum() + pred_bin.sum() + 1e-8)

# ===== 寬鬆版 IoU / Dice =====
from skimage.morphology import dilation, disk

tol = 2
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

# ===== 重疊區域可視化 =====
overlay = np.zeros((256, 256, 3), dtype=np.float32)
overlay[..., 1] = gt.astype(np.float32)        # green = GT
overlay[..., 0] = pred_bin.astype(np.float32)  # red = Pred

overlay_tol = np.zeros((256, 256, 3), dtype=np.float32)
overlay_tol[..., 1] = gt_dilated.astype(np.float32)
overlay_tol[..., 0] = pred_dilated.astype(np.float32)

# ===== 畫圖：2 x 4 =====
fig, ax = plt.subplots(2, 4, figsize=(12, 9))

# 1. 強化後原圖
ax[0, 0].imshow(img_enhanced, cmap="gray")
ax[0, 0].set_title("強化後原圖")
ax[0, 0].axis("off")

# 2. Ground Truth mask
ax[0, 1].imshow(gt, cmap="gray")
ax[0, 1].set_title("ground truth mask")
ax[0, 1].axis("off")

# 3. 模型熱圖
ax[0, 2].imshow(pred, cmap="hot")
ax[0, 2].set_title("熱圖")
ax[0, 2].axis("off")

# 4. Prediction binary mask
ax[0, 3].imshow(pred_bin, cmap="gray")
ax[0, 3].set_title("prediction binary mask")
ax[0, 3].axis("off")

# 5. 強化後影像 + 論文結果（GT）
ax[1, 0].imshow(img_enhanced, cmap="gray")
ax[1, 0].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
ax[1, 0].set_title("強化後影像 + 論文結果(GT)")
ax[1, 0].axis("off")

# 6. 強化後影像 + 你的結果（Pred）
ax[1, 1].imshow(img_enhanced, cmap="gray")
ax[1, 1].contour(pred_bin, levels=[0.5], colors="red", linewidths=1.0)
ax[1, 1].set_title("強化後影像 + 我的結果(Pred)")
ax[1, 1].axis("off")

# 7. 強化後影像 + 論文結果 + 我的結果
ax[1, 2].imshow(img_enhanced, cmap="gray")
ax[1, 2].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
ax[1, 2].contour(pred_bin, levels=[0.5], colors="red", linewidths=1.0)
ax[1, 2].set_title("強化後影像 + GT(green) + Pred(red)")
ax[1, 2].axis("off")

# 8. Overlap
ax[1, 3].imshow(overlay)
ax[1, 3].set_title(f"Overlap\nIoU={iou:.4f}, Dice={dice:.4f}")
ax[1, 3].axis("off")

plt.tight_layout()
plt.show()

# ===== 額外顯示寬鬆版 overlap =====
plt.figure(figsize=(6, 6))
plt.imshow(overlay_tol)
plt.title(f"Relaxed Overlap, tol={tol}px\nIoU={iou_tol:.4f}, Dice={dice_tol:.4f}")
plt.axis("off")
plt.show()
