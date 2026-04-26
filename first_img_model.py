import os
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Circle
from skimage.morphology import dilation, disk, binary_closing, remove_small_objects
from skimage.transform import hough_circle, hough_circle_peaks

# =========================
# 路徑設定
# =========================
dev_images_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
model_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"

# 輸出資料夾
output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\single_result_output"
os.makedirs(output_dir, exist_ok=True)

# 測試第幾張
idx = 0

# 圓偵測參數（可調）
R_MIN = 5
R_MAX = 80
R_STEP = 1
MAX_CIRCLES = 20
THRESHOLD_RATIO = 0.25   # pred > 0.25 * pred.max()
MIN_OBJ_SIZE = 20

# =========================
# 小工具：去掉太接近的重複圓
# =========================
def deduplicate_circles(cx, cy, radii, scores, center_dist_thresh=12, radius_diff_thresh=6):
    keep = []
    for i in range(len(radii)):
        duplicated = False
        for j in keep:
            d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            dr = abs(radii[i] - radii[j])
            if d < center_dist_thresh and dr < radius_diff_thresh:
                duplicated = True
                break
        if not duplicated:
            keep.append(i)
    return cx[keep], cy[keep], radii[keep], scores[keep]

# =========================
# 讀模型
# =========================
model = tf.keras.models.load_model(model_path, compile=False)

# =========================
# 讀資料
# =========================
with h5py.File(dev_images_path, "r") as f:
    img = f["input_images"][idx]
    mask = f["target_masks"][idx]

# =========================
# 前處理（模型輸入）
# =========================
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

# =========================
# 強化影像（只顯示用）
# =========================
try:
    from skimage import exposure
    img_enhanced = exposure.equalize_adapthist(img_norm, clip_limit=0.03)
except ModuleNotFoundError:
    print("沒有安裝 scikit-image 的 exposure，改用簡單 contrast stretch")
    p2, p98 = np.percentile(img_norm, (2, 98))
    img_enhanced = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

# =========================
# GT 與 Prediction Binary
# =========================
gt = mask > 0

threshold = THRESHOLD_RATIO * pred.max()
pred_bin = pred > threshold
pred_bin = binary_closing(pred_bin, footprint=disk(1))
pred_bin = remove_small_objects(pred_bin, min_size=MIN_OBJ_SIZE)

# =========================
# IoU / Dice
# =========================
intersection = np.logical_and(gt, pred_bin).sum()
union = np.logical_or(gt, pred_bin).sum()

iou = intersection / (union + 1e-8)
dice = 2 * intersection / (gt.sum() + pred_bin.sum() + 1e-8)

tol = 2
se = disk(tol)

gt_dilated = dilation(gt, se)
pred_dilated = dilation(pred_bin, se)

intersection_tol = np.logical_and(gt_dilated, pred_dilated).sum()
union_tol = np.logical_or(gt_dilated, pred_dilated).sum()

iou_tol = intersection_tol / (union_tol + 1e-8)
dice_tol = 2 * intersection_tol / (gt_dilated.sum() + pred_dilated.sum() + 1e-8)

# =========================
# 用 Hough Circle 找圓
# =========================
hough_radii = np.arange(R_MIN, R_MAX + 1, R_STEP)
hough_res = hough_circle(pred_bin.astype(np.uint8), hough_radii)

accums, cx, cy, radii = hough_circle_peaks(
    hough_res,
    hough_radii,
    total_num_peaks=MAX_CIRCLES,
    min_xdistance=12,
    min_ydistance=12,
    normalize=True
)

# 去重複
cx, cy, radii, accums = deduplicate_circles(cx, cy, radii, accums)

# =========================
# 整理結果表
# =========================
results = []
for i in range(len(radii)):
    results.append({
        "id": i + 1,
        "x_center": float(cx[i]),
        "y_center": float(cy[i]),
        "radius": float(radii[i]),
        "diameter": float(radii[i] * 2),
        "score": float(accums[i])
    })

results_df = pd.DataFrame(results)

print("\n==============================")
print(f"第 {idx} 張圖")
print(f"找到的隕石坑數量 = {len(results_df)}")
print("==============================")

if len(results_df) > 0:
    print(results_df)
else:
    print("沒有找到任何圓。")

# 存 CSV
csv_path = os.path.join(output_dir, f"craters_info_{idx:05d}.csv")
results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n已儲存 CSV：{csv_path}")

# =========================
# 重疊區域圖
# =========================
overlay = np.zeros((256, 256, 3), dtype=np.float32)
overlay[..., 1] = gt.astype(np.float32)        # green = GT
overlay[..., 0] = pred_bin.astype(np.float32)  # red = Pred

# =========================
# 畫圖
# =========================
fig, ax = plt.subplots(2, 4, figsize=(20, 10))

# 1 強化後原圖
ax[0, 0].imshow(img_enhanced, cmap="gray")
ax[0, 0].set_title("image")
ax[0, 0].axis("off")

# 2 GT mask
ax[0, 1].imshow(gt, cmap="gray")
ax[0, 1].set_title("ground truth mask")
ax[0, 1].axis("off")

# 3 熱圖
ax[0, 2].imshow(pred, cmap="hot")
ax[0, 2].set_title("熱圖")
ax[0, 2].axis("off")

# 4 binary mask
ax[0, 3].imshow(pred_bin, cmap="gray")
ax[0, 3].set_title("prediction binary mask")
ax[0, 3].axis("off")

# 5 強化圖 + GT
ax[1, 0].imshow(img_enhanced, cmap="gray")
ax[1, 0].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
ax[1, 0].set_title("強化後影像 + 論文結果(GT)")
ax[1, 0].axis("off")

# 6 強化圖 + 你找到的圓
ax[1, 1].imshow(img_enhanced, cmap="gray")
for i in range(len(radii)):
    circ = Circle((cx[i], cy[i]), radii[i], fill=False, edgecolor="red", linewidth=2)
    ax[1, 1].add_patch(circ)
    ax[1, 1].text(cx[i], cy[i], str(i + 1), color="yellow", fontsize=10, ha="center", va="center")
ax[1, 1].set_title(f"我的結果：找到 {len(radii)} 個隕石坑")
ax[1, 1].axis("off")

# 7 強化圖 + GT + 我找到的圓
ax[1, 2].imshow(img_enhanced, cmap="gray")
ax[1, 2].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
for i in range(len(radii)):
    circ = Circle((cx[i], cy[i]), radii[i], fill=False, edgecolor="red", linewidth=2)
    ax[1, 2].add_patch(circ)
    ax[1, 2].text(cx[i], cy[i], str(i + 1), color="yellow", fontsize=10, ha="center", va="center")
ax[1, 2].set_title("強化後影像 + GT(green) + 我的結果(red)")
ax[1, 2].axis("off")

# 8 overlap
ax[1, 3].imshow(overlay)
ax[1, 3].set_title(f"Overlap\nIoU={iou:.4f}, Dice={dice:.4f}")
ax[1, 3].axis("off")

plt.tight_layout()

# 存圖片
fig_path = os.path.join(output_dir, f"result_plot_{idx:05d}.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"已儲存結果圖：{fig_path}")

plt.show()
