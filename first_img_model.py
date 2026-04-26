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

output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\single_result_output"
os.makedirs(output_dir, exist_ok=True)

# 測試第幾張
idx = 0


# =========================
# 參數設定
# =========================

THRESHOLD_RATIOS = [0.13, 0.16, 0.18, 0.22]

MIN_OBJ_SIZE = 10
MIN_ARC_POINTS = 6

R_MIN = 4
R_MAX = 180

MIN_BBOX_SIZE = 8
MAX_RESIDUAL = 12

MERGE_CENTER_DIST = 35
MERGE_RADIUS_DIFF = 35


# ===== 左上邊界大圓專用參數 =====
# 這組比較保守，避免左上大圓被中間 crater 拉歪
LEFT_TOP_X_LIMIT = 105
LEFT_TOP_Y_LIMIT = 170

LEFT_TOP_FIRST_X = 22
LEFT_TOP_FIRST_Y = 95

LEFT_TOP_R_MIN = 60
LEFT_TOP_R_MAX = 120

LEFT_TOP_MAX_RESIDUAL = 10
LEFT_TOP_RING_TOL = 3.0


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

        # 允許圓心在圖片外，因為邊界 crater 可能只露出一部分
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

                # 如果新的圓品質更好，就替換
                if (c["arc_area"] > m["arc_area"]) and (c["residual"] < m["residual"] * 1.5):
                    m.update(c)

                break

        if not duplicated:
            merged.append(c)

    return merged


# =========================
# 工具：移除在大圓內部的小假圓
# =========================
def remove_nested_small_false_circles(circles):
    """
    移除落在大 crater 內部的小假圓。
    這裡條件有加強，讓像之前 6、7 那種內部假圓比較容易被濾掉。
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

            # 只用比自己大很多的圓判斷
            if br < r * 2.2:
                continue

            d = np.sqrt((xc - bx) ** 2 + (yc - by) ** 2)

            # 加強版：中心落在大圓內部，就比較容易被視為假小圓
            if d < br * 0.80 and r < br * 0.45:
                is_false_nested = True
                break

        if not is_false_nested:
            keep.append(c)

    return keep


# =========================
# 工具：補抓左上邊界大圓
# =========================
def add_left_top_boundary_big_circle(pred_bin, circles):
    """
    專門補抓左上角邊界大圓。
    改良版做法：
    1) 先只取「很靠上 / 很靠左」的點做粗略 fit
    2) 再根據粗略 fit，挑選貼近該圓的點做 refined fit
    """
    if circles is None:
        circles = []

    ys, xs = np.where(pred_bin)

    # -----------------------------
    # Step 1：先只取左上區域，而且只取很靠上 or 很靠左的點
    # 避免下面那顆大 crater 把圓心拉歪
    # -----------------------------
    sel1 = (
        (xs < LEFT_TOP_X_LIMIT) &
        (ys < LEFT_TOP_Y_LIMIT) &
        (
            (xs < LEFT_TOP_FIRST_X) |
            (ys < LEFT_TOP_FIRST_Y)
        )
    )

    xs1 = xs[sel1]
    ys1 = ys[sel1]

    print("\n===== Left-top boundary debug =====")
    print("coarse candidate points =", len(xs1))

    if len(xs1) < 12:
        print("原因：左上候選點太少")
        print("==================================\n")
        return circles

    points1 = np.column_stack([xs1, ys1])

    fit1 = fit_circle_least_squares(points1)
    if fit1 is None:
        print("原因：第一次 coarse fit 失敗")
        print("==================================\n")
        return circles

    xc1, yc1, r1, residual1 = fit1
    print(f"coarse fit -> xc={xc1:.2f}, yc={yc1:.2f}, r={r1:.2f}, residual={residual1:.2f}")

    if r1 < LEFT_TOP_R_MIN or r1 > LEFT_TOP_R_MAX:
        print("原因：coarse fit 半徑不合理")
        print("==================================\n")
        return circles

    # -----------------------------
    # Step 2：refined fit
    # 只保留「貼近粗略圓」的點，再重 fit 一次
    # -----------------------------
    dist_all = np.sqrt((xs - xc1) ** 2 + (ys - yc1) ** 2)

    sel2 = (
        (xs < LEFT_TOP_X_LIMIT) &
        (ys < LEFT_TOP_Y_LIMIT) &
        (np.abs(dist_all - r1) < LEFT_TOP_RING_TOL)
    )

    xs2 = xs[sel2]
    ys2 = ys[sel2]

    print("refined candidate points =", len(xs2))

    if len(xs2) >= 12:
        points2 = np.column_stack([xs2, ys2])
        fit2 = fit_circle_least_squares(points2)

        if fit2 is not None:
            xc, yc, r, residual = fit2
        else:
            xc, yc, r, residual = xc1, yc1, r1, residual1
    else:
        xc, yc, r, residual = xc1, yc1, r1, residual1

    print(f"refined fit -> xc={xc:.2f}, yc={yc:.2f}, r={r:.2f}, residual={residual:.2f}")

    # -----------------------------
    # Step 3：最後條件限制
    # -----------------------------
    if r < LEFT_TOP_R_MIN or r > LEFT_TOP_R_MAX:
        print("原因：refined fit 半徑不合理")
        print("==================================\n")
        return circles

    # 左上邊界大圓通常圓心不會跑到太右太下
    if xc > 90 or yc > 150:
        print("原因：圓心位置太偏右/偏下，不像左上邊界大圓")
        print("==================================\n")
        return circles

    if residual > LEFT_TOP_MAX_RESIDUAL:
        print("原因：residual 太大")
        print("==================================\n")
        return circles

    # 如果跟現有某個圓太像，就不要重複加
    for c in circles:
        d = np.sqrt((xc - c["x_center"]) ** 2 + (yc - c["y_center"]) ** 2)
        dr = abs(r - c["radius"])

        if d < 20 and dr < 20:
            print("原因：和既有圓太接近，不重複加入")
            print("==================================\n")
            return circles

    new_circle = {
        "x_center": float(xc),
        "y_center": float(yc),
        "radius": float(r),
        "diameter": float(r * 2),
        "arc_area": int(len(xs2)) if len(xs2) >= 12 else int(len(xs1)),
        "residual": float(residual),
        "method": "left_top_boundary_arc",
        "threshold_ratio": None,
        "threshold": None
    }

    circles.append(new_circle)

    print("成功：已補上左上邊界大圓")
    print("==================================\n")

    return circles


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
# 強化影像：只顯示用
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

    # 保守版：closing 用 disk(2)，不要 dilation
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
# 合併 + 補抓邊界大圓 + 移除假小圓
# =========================
final_circles = merge_similar_circles(all_arc_circles)

# 補抓左上邊界大圓
final_circles = add_left_top_boundary_big_circle(pred_bin, final_circles)

# 再合併一次，避免重複
final_circles = merge_similar_circles(final_circles)

# 移除在大圓內的小假圓
final_circles = remove_nested_small_false_circles(final_circles)

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

print("\n==============================")
print(f"Image index = {idx}")
print(f"Detected craters = {len(results_df)}")
print(f"IoU = {iou:.4f}, Dice = {dice:.4f}")
print(f"Relaxed IoU = {iou_tol:.4f}, Relaxed Dice = {dice_tol:.4f}")
print("==============================")

if len(results_df) > 0:
    print(results_df)
else:
    print("No circles detected.")

csv_path = os.path.join(output_dir, f"arcfit_boundary_fixed_craters_info_{idx:05d}.csv")
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

fig_path = os.path.join(output_dir, f"arcfit_boundary_fixed_result_plot_{idx:05d}.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Figure saved: {fig_path}")

plt.show()
