import os
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Circle
from skimage.morphology import dilation, disk, binary_closing, remove_small_objects
from skimage.measure import label, regionprops


# =========================
# 路徑設定
# =========================
dev_images_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
model_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"

output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\report_analysis_output"
os.makedirs(output_dir, exist_ok=True)

# 測試第幾張
idx = 0


# =========================
# 參數設定
# =========================
THRESHOLD_RATIOS = [0.12, 0.13, 0.16, 0.18, 0.22, 0.26]

MIN_OBJ_SIZE = 6
MIN_ARC_POINTS = 5

R_MIN = 4
R_MAX = 180

MIN_BBOX_SIZE = 5
MAX_RESIDUAL = 10

MERGE_CENTER_DIST = 35
MERGE_RADIUS_DIFF = 35


# =========================
# 工具：從一段弧線 fitting 圓
# =========================
def fit_circle_least_squares(points):
    """
    points: shape = (N, 2), each row is [x, y]
    return: xc, yc, r, residual
    """
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]

    # x^2 + y^2 + D*x + E*y + F = 0
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)

    try:
        D, E, F = np.linalg.lstsq(A, B, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    xc = -D / 2
    yc = -E / 2

    r2 = xc**2 + yc**2 - F

    if r2 <= 0:
        return None

    r = np.sqrt(r2)

    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    residual = np.mean(np.abs(distances - r))

    return xc, yc, r, residual


# =========================
# 工具：從 pred_bin 裡每段弧線 fitting 圓
# =========================
def arc_fitting_from_mask(pred_bin):
    labeled = label(pred_bin)
    regions = regionprops(labeled)

    circles = []

    for region in regions:
        if region.area < MIN_ARC_POINTS:
            continue

        coords = region.coords

        # region.coords 是 [row, col] = [y, x]
        y = coords[:, 0]
        x = coords[:, 1]

        bbox_h = region.bbox[2] - region.bbox[0]
        bbox_w = region.bbox[3] - region.bbox[1]

        # 太短的弧線不要單獨 fitting，避免短雜訊變成假圓
        if max(bbox_h, bbox_w) < MIN_BBOX_SIZE:
            continue

        points = np.column_stack([x, y])

        fit = fit_circle_least_squares(points)
        if fit is None:
            continue

        xc, yc, r, residual = fit

        if r < R_MIN or r > R_MAX:
            continue

        # 允許圓心稍微在圖片外，因為邊界 crater 可能只露出一部分
        if xc < -200 or xc > 456 or yc < -200 or yc > 456:
            continue

        if residual > MAX_RESIDUAL:
            continue

        circles.append({
            "x_center": float(xc),
            "y_center": float(yc),
            "radius": float(r),
            "diameter": float(r * 2),
            "arc_area": int(region.area),
            "residual": float(residual),
            "method": "arc_fit"
        })

    return circles


# =========================
# 工具：合併太相似的圓
# =========================
def merge_similar_circles(circles):
    if circles is None or len(circles) == 0:
        return []

    # 優先保留弧線點多、residual 小的圓
    circles = sorted(circles, key=lambda c: (-c["arc_area"], c["residual"]))

    merged = []

    for c in circles:
        xc = c["x_center"]
        yc = c["y_center"]
        r = c["radius"]

        duplicated = False

        for m in merged:
            d = np.sqrt((xc - m["x_center"]) ** 2 + (yc - m["y_center"]) ** 2)
            dr = abs(r - m["radius"])

            if d < MERGE_CENTER_DIST and dr < MERGE_RADIUS_DIFF:
                duplicated = True

                if (c["arc_area"] > m["arc_area"]) and (c["residual"] < m["residual"] * 1.5):
                    m.update(c)

                break

        if not duplicated:
            merged.append(c)

    return merged


# =========================
# 工具：移除大圓內部的小假圓
# =========================
def remove_nested_small_false_circles(circles):
    """
    移除落在大 crater 內部的假小圓。
    但如果小圓接近大圓邊界 rim，則保留。
    """
    if circles is None or len(circles) == 0:
        return []

    keep = []

    for i, c in enumerate(circles):
        xc = c["x_center"]
        yc = c["y_center"]
        r = c["radius"]

        is_false_nested = False

        for j, big in enumerate(circles):
            if i == j:
                continue

            bx = big["x_center"]
            by = big["y_center"]
            br = big["radius"]

            # 只拿明顯比較大的圓來判斷
            if br < r * 2.5:
                continue

            d = np.sqrt((xc - bx) ** 2 + (yc - by) ** 2)

            # 小圓中心到大圓邊界的距離
            rim_dist = abs(d - br)

            # 如果小圓中心接近大圓 rim，可能是真實附著在邊界上的小 crater，不刪
            if rim_dist < max(4, 0.35 * r):
                continue

            # 如果小圓很深地落在大圓內部，才刪掉
            if d < br * 0.72 and r < br * 0.35:
                is_false_nested = True
                break

        if not is_false_nested:
            keep.append(c)

    return keep


# =========================
# 讀模型
# =========================
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded.")


# =========================
# 讀資料
# =========================
with h5py.File(dev_images_path, "r") as f:
    img = f["input_images"][idx]
    mask = f["target_masks"][idx]


# =========================
# 前處理：模型輸入
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
# 強化影像：只顯示用，不餵給模型
# =========================
try:
    from skimage import exposure
    img_enhanced = exposure.equalize_adapthist(img_norm, clip_limit=0.03)
except ModuleNotFoundError:
    print("No skimage.exposure, use contrast stretch.")
    p2, p98 = np.percentile(img_norm, (2, 98))
    img_enhanced = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)


# =========================
# Ground Truth
# =========================
gt = mask > 0


# =========================
# 多 threshold 產生 pred_bin + arc fitting
# =========================
all_arc_circles = []
all_pred_bins = []

for ratio in THRESHOLD_RATIOS:
    threshold = ratio * pred.max()

    temp_bin = pred > threshold

    # closing 補一點點斷裂，但不要 dilation，避免假圓變多
    temp_bin = binary_closing(temp_bin, footprint=disk(2))
    temp_bin = remove_small_objects(temp_bin, min_size=MIN_OBJ_SIZE)

    temp_circles = arc_fitting_from_mask(temp_bin)

    for c in temp_circles:
        c["threshold_ratio"] = ratio
        c["threshold"] = float(threshold)

    all_arc_circles.extend(temp_circles)
    all_pred_bins.append(temp_bin)

# 合併不同 threshold 的 binary mask，作為顯示與 IoU 使用
pred_bin = np.logical_or.reduce(all_pred_bins)


# =========================
# 報告分析用：只使用泛用 arc fitting，不做人工邊界補抓
# =========================
final_circles = merge_similar_circles(all_arc_circles)

# 移除大圓內部假小圓
final_circles = remove_nested_small_false_circles(final_circles)

# 最後再合併一次
final_circles = merge_similar_circles(final_circles)

cx = np.array([c["x_center"] for c in final_circles])
cy = np.array([c["y_center"] for c in final_circles])
radii = np.array([c["radius"] for c in final_circles])


# =========================
# IoU / Dice
# =========================
intersection = np.logical_and(gt, pred_bin).sum()
union = np.logical_or(gt, pred_bin).sum()

iou = intersection / (union + 1e-8)
dice = 2 * intersection / (gt.sum() + pred_bin.sum() + 1e-8)


# =========================
# Relaxed IoU / Dice
# =========================
tol = 2
se = disk(tol)

gt_dilated = dilation(gt, se)
pred_dilated = dilation(pred_bin, se)

intersection_tol = np.logical_and(gt_dilated, pred_dilated).sum()
union_tol = np.logical_or(gt_dilated, pred_dilated).sum()

iou_tol = intersection_tol / (union_tol + 1e-8)
dice_tol = 2 * intersection_tol / (gt_dilated.sum() + pred_dilated.sum() + 1e-8)


# =========================
# 整理結果表
# =========================
results = []

for i, c in enumerate(final_circles):
    results.append({
        "id": i + 1,
        "x_center": c["x_center"],
        "y_center": c["y_center"],
        "radius": c["radius"],
        "diameter": c["diameter"],
        "arc_area": c.get("arc_area", None),
        "residual": c.get("residual", None),
        "method": c.get("method", None),
        "threshold_ratio": c.get("threshold_ratio", None),
        "threshold": c.get("threshold", None)
    })

results_df = pd.DataFrame(results)


# =========================
# 印出結果
# =========================
print("\n==============================")
print(f"Image index = {idx}")
print("Post-processing mode = arc fitting only, no manual boundary correction")
print(f"Detected craters = {len(results_df)}")
print(f"IoU = {iou:.4f}, Dice = {dice:.4f}")
print(f"Relaxed IoU = {iou_tol:.4f}, Relaxed Dice = {dice_tol:.4f}")
print("==============================")

if len(results_df) > 0:
    print(results_df)
else:
    print("No circles detected.")


# =========================
# 存 CSV
# =========================
csv_path = os.path.join(output_dir, f"report_arcfit_craters_info_{idx:05d}.csv")
results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"CSV saved: {csv_path}")


# =========================
# Overlay
# =========================
overlay = np.zeros((256, 256, 3), dtype=np.float32)
overlay[..., 1] = gt.astype(np.float32)
overlay[..., 0] = pred_bin.astype(np.float32)


# =========================
# 畫圖
# =========================
fig, ax = plt.subplots(2, 4, figsize=(20, 10))

# 1 image
ax[0, 0].imshow(img_enhanced, cmap="gray")
ax[0, 0].set_title("image")
ax[0, 0].axis("off")

# 2 GT mask
ax[0, 1].imshow(gt, cmap="gray")
ax[0, 1].set_title("ground truth mask")
ax[0, 1].axis("off")

# 3 heatmap
ax[0, 2].imshow(pred, cmap="hot")
ax[0, 2].set_title("heatmap")
ax[0, 2].axis("off")

# 4 prediction binary mask
ax[0, 3].imshow(pred_bin, cmap="gray")
ax[0, 3].set_title("prediction binary mask")
ax[0, 3].axis("off")

# 5 GT
ax[1, 0].imshow(img_enhanced, cmap="gray")
ax[1, 0].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
ax[1, 0].set_title("GT")
ax[1, 0].axis("off")

# 6 Arc fitting result
ax[1, 1].imshow(img_enhanced, cmap="gray")
for i in range(len(radii)):
    circ = Circle((cx[i], cy[i]), radii[i], fill=False, edgecolor="red", linewidth=2)
    ax[1, 1].add_patch(circ)
    ax[1, 1].text(
        cx[i],
        cy[i],
        str(i + 1),
        color="yellow",
        fontsize=10,
        ha="center",
        va="center"
    )

ax[1, 1].set_title(f"Arc fitting: {len(radii)} craters")
ax[1, 1].axis("off")

# 7 GT + ArcFit
ax[1, 2].imshow(img_enhanced, cmap="gray")
ax[1, 2].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)

for i in range(len(radii)):
    circ = Circle((cx[i], cy[i]), radii[i], fill=False, edgecolor="red", linewidth=2)
    ax[1, 2].add_patch(circ)
    ax[1, 2].text(
        cx[i],
        cy[i],
        str(i + 1),
        color="yellow",
        fontsize=10,
        ha="center",
        va="center"
    )

ax[1, 2].set_title("GT(green) + ArcFit(red)")
ax[1, 2].axis("off")

# 8 Overlap
ax[1, 3].imshow(overlay)
ax[1, 3].set_title(f"Overlap\nIoU={iou:.4f}, Dice={dice:.4f}")
ax[1, 3].axis("off")

plt.tight_layout()

fig_path = os.path.join(output_dir, f"report_arcfit_result_plot_{idx:05d}.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Figure saved: {fig_path}")

plt.show()


# =========================
# 額外 relaxed overlap 圖
# =========================
overlay_tol = np.zeros((256, 256, 3), dtype=np.float32)
overlay_tol[..., 1] = gt_dilated.astype(np.float32)
overlay_tol[..., 0] = pred_dilated.astype(np.float32)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_tol)
plt.title(f"Relaxed Overlap, tol={tol}px\nIoU={iou_tol:.4f}, Dice={dice_tol:.4f}")
plt.axis("off")

relaxed_path = os.path.join(output_dir, f"report_relaxed_overlap_{idx:05d}.png")
plt.savefig(relaxed_path, dpi=200, bbox_inches="tight")
print(f"Relaxed overlap saved: {relaxed_path}")

plt.show()
