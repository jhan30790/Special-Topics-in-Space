import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource

# =============================================================
# 路徑設定
# =============================================================
DEEPMOON_DIR = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master"
sys.path.insert(0, DEEPMOON_DIR)
import utils.template_match_target as tmt

dev_images_path  = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
dev_craters_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_craters.hdf5"
model_path       = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"
output_dir       = r"C:\Users\jhan3\OneDrive\桌面\大二專題\result_global_eval_2"
os.makedirs(output_dir, exist_ok=True)

# =============================================================
# 參數
# =============================================================
NUM_IMAGES      = 100
IMG_SIZE        = 256
MINRAD          = 5
MAXRAD          = 40
TARGET_THRESH   = 0.1
LONGLAT_THRESH2 = 1.8
RAD_THRESH      = 1.0
TEMPLATE_THRESH = 0.35

# 月球半徑（km），用來把 km 轉成度
MOON_RADIUS_KM  = 1737.4

# =============================================================
# 工具函式
# =============================================================
def pixel_to_longlat(x_pix, y_pix, longlat_bounds, img_size=256):
    """像素座標 → 月面經緯度"""
    lon_min, lat_min, lon_max, lat_max = longlat_bounds
    lon = lon_min + (x_pix / img_size) * (lon_max - lon_min)
    lat = lat_min + (y_pix / img_size) * (lat_max - lat_min)
    return lon, lat

def longlat_to_pixel(lon, lat, longlat_bounds, img_size=256):
    """月面經緯度 → 像素座標"""
    lon_min, lat_min, lon_max, lat_max = longlat_bounds
    x = (lon - lon_min) / (lon_max - lon_min) * img_size
    y = (lat - lat_min) / (lat_max - lat_min) * img_size
    return x, y

def km_to_deg(diameter_km):
    """直徑 km → 度（用於去重複的距離比較）"""
    return (diameter_km / (2 * np.pi * MOON_RADIUS_KM)) * 360

def pix_radius_to_deg(r_pix, longlat_bounds, img_size=256):
    """像素半徑 → 度"""
    lon_min, _, lon_max, lat_max = longlat_bounds
    deg_per_pix = (lon_max - lon_min) / img_size
    return r_pix * deg_per_pix

def global_deduplicate(pred_world, longlat_thresh2=1.8, rad_thresh=1.0):
    """
    跨圖去重複：如果兩個預測的經緯度距離 < threshold 且半徑相近，
    保留一個，刪掉另一個。
    pred_world: list of (lon, lat, radius_deg, img_idx)
    """
    if len(pred_world) == 0:
        return []

    arr = np.array([(p[0], p[1], p[2]) for p in pred_world])
    keep = np.ones(len(arr), dtype=bool)

    for i in range(len(arr)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(arr)):
            if not keep[j]:
                continue
            dlon = arr[i, 0] - arr[j, 0]
            dlat = arr[i, 1] - arr[j, 1]
            dist2 = dlon**2 + dlat**2
            r_avg = (arr[i, 2] + arr[j, 2]) / 2
            # 用跟 template_match_t 一樣的邏輯
            if dist2 < longlat_thresh2 * r_avg**2:
                keep[j] = False  # 保留 i，刪掉 j

    return [pred_world[i] for i in range(len(pred_world)) if keep[i]]

def make_z_surface(img_norm):
    dem_smooth = gaussian_filter(img_norm, sigma=1.5)
    p1, p99 = np.percentile(dem_smooth, (1, 99))
    dem_clip = np.clip(dem_smooth, p1, p99)
    return (dem_clip - p1) / (p99 - p1 + 1e-8)

def circle_points_3d(z_surface, xc, yc, r, n_points=200, z_offset=0.03):
    h, w = z_surface.shape
    theta = np.linspace(0, 2 * np.pi, n_points)
    xs = xc + r * np.cos(theta)
    ys = yc + r * np.sin(theta)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        return np.array([]), np.array([]), np.array([])
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    zs = z_surface[yi, xi] + z_offset
    return xs, ys, zs

def draw_3d_surface(ax, z_surface, craters_gt, craters_pred, h, w):
    Y, X = np.mgrid[0:h, 0:w]
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(z_surface, cmap=plt.cm.gray,
                   vert_exag=1, blend_mode="soft")
    ax.plot_surface(X, Y, z_surface, facecolors=rgb,
                    linewidth=0, antialiased=True, shade=False, alpha=0.85)
    for (gx, gy, gr) in craters_gt:
        xs, ys, zs = circle_points_3d(z_surface, gx, gy, gr, z_offset=0.05)
        if len(xs) > 0:
            ax.plot(xs, ys, zs, color="lime", linewidth=2)
    for (px, py, pr) in craters_pred:
        xs, ys, zs = circle_points_3d(z_surface, px, py, pr, z_offset=0.03)
        if len(xs) > 0:
            ax.plot(xs, ys, zs, color="red", linewidth=2)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_zlim(0, np.max(z_surface) + 0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.view_init(elev=35, azim=-60)

# =============================================================
# 載入模型
# =============================================================
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded.\n")

# =============================================================
# 第一輪：跑模型，收集所有預測的月面座標
# =============================================================
print("=== 第一輪：模型推論 + 收集月面座標 ===")

all_data = []        # 每張圖的完整資料
pred_world_all = []  # 所有圖的預測，轉成月面座標 (lon, lat, r_deg, img_idx)
gt_world_all   = []  # 所有圖的 GT 月面座標 (lon, lat, r_deg)

with h5py.File(dev_images_path,  "r") as f_img, \
     h5py.File(dev_craters_path, "r") as f_crt:

    for idx in range(NUM_IMAGES):
        img_key = f"img_{idx:05d}"

        # 讀圖
        img = f_img["input_images"][idx].astype("float32")
        longlat_bounds = f_img["longlat_bounds"][img_key][:]

        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        x_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # 模型推論
        pred_raw = model.predict(x_input, verbose=0)
        if pred_raw.ndim == 4:
            pred = pred_raw[0, :, :, 0]
        elif pred_raw.ndim == 3:
            pred = pred_raw[0, :, :]

        # 後處理：得到像素座標
        pred_coords_pix = tmt.template_match_t(
            pred.copy(),
            minrad=MINRAD, maxrad=MAXRAD,
            longlat_thresh2=LONGLAT_THRESH2,
            rad_thresh=RAD_THRESH,
            template_thresh=TEMPLATE_THRESH,
            target_thresh=TARGET_THRESH
        )

        # 預測像素座標 → 月面座標
        for (px, py, pr) in pred_coords_pix:
            lon, lat = pixel_to_longlat(px, py, longlat_bounds)
            r_deg = pix_radius_to_deg(pr, longlat_bounds)
            pred_world_all.append((lon, lat, r_deg, idx))

        # 讀 GT
        try:
            block = f_crt[img_key]['block0_values'][...]
            if block.ndim == 1:
                block = block.reshape(1, -1)
            if block.shape[0] == 6 and block.shape[1] != 6:
                block = block.T
            if block.shape[1] >= 6:
                gt_lon  = block[:, 2]
                gt_lat  = block[:, 1]
                gt_diam_km = block[:, 0]
                csv_coords = np.column_stack([block[:, 3], block[:, 4], block[:, 5] / 2.0])
                for i in range(len(gt_lon)):
                    r_deg = km_to_deg(gt_diam_km[i] / 2)
                    gt_world_all.append((gt_lon[i], gt_lat[i], r_deg))
            else:
                csv_coords = np.empty((0, 3))
        except KeyError:
            csv_coords = np.empty((0, 3))

        # 顯示用圖
        p2, p98 = np.percentile(img_norm, (2, 98))
        img_disp = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

        all_data.append({
            "img_norm":      img_norm,
            "img_disp":      img_disp,
            "pred":          pred,
            "pred_pix":      pred_coords_pix,   # 原始像素預測（去重複前）
            "csv_coords":    csv_coords,
            "longlat_bounds": longlat_bounds,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{NUM_IMAGES}] 預測完成，本圖預測數：{len(pred_coords_pix)}")

print(f"\n第一輪完成。總預測數（去重複前）：{len(pred_world_all)}")

# =============================================================
# 全局去重複
# =============================================================
print("\n=== 全局去重複 ===")
pred_world_dedup = global_deduplicate(pred_world_all)
print(f"去重複前：{len(pred_world_all)}，去重複後：{len(pred_world_dedup)}")

# GT 也去重複（不同圖可能有相同 GT 坑）
gt_world_dedup = global_deduplicate(
    [(lon, lat, r, 0) for (lon, lat, r) in gt_world_all]
)
gt_world_dedup = [(lon, lat, r) for (lon, lat, r, _) in gt_world_dedup]
print(f"GT 去重複前：{len(gt_world_all)}，去重複後：{len(gt_world_dedup)}")

# =============================================================
# 全局 TP/FP/FN 比對（在像素空間做，GT經緯度轉回各圖像素）
# =============================================================
print("\n=== 全局比對 ===")

# 把去重複後的預測和GT都對應回各圖像素座標來比對
# 每個 GT 坑找它落在哪些圖裡，轉成像素座標後比對

# 先建立去重複後預測的像素座標（按圖分組，已在第二輪做過）
# 這裡重新建立全局 matched 集合

global_matched_gt   = set()  # gt_world_dedup 的 index
global_matched_pred = set()  # pred_world_dedup 的 index

# 對每張圖，在像素空間做比對
for idx in range(NUM_IMAGES):
    data           = all_data[idx]
    csv_coords     = data["csv_coords"]      # 該圖 GT 像素座標
    longlat_bounds = data["longlat_bounds"]

    if len(csv_coords) == 0:
        continue

    # 該圖去重複後的預測（像素座標）
    pred_pix_local = []
    pred_global_idx = []
    for gi, (lon, lat, r_deg, img_idx) in enumerate(pred_world_dedup):
        if img_idx != idx:
            continue
        px, py = longlat_to_pixel(lon, lat, longlat_bounds)
        lon_min, _, lon_max, _ = longlat_bounds
        pr = r_deg / ((lon_max - lon_min) / IMG_SIZE)
        pred_pix_local.append((px, py, pr))
        pred_global_idx.append(gi)

    # GT 像素座標已在 csv_coords，建立全局 GT index 的映射
    # GT 全局 index：用 gt_world_dedup 對應
    # 簡化：直接用逐張像素比對算全局 TP
    for pi, (px, py, pr) in enumerate(pred_pix_local):
        g_pi = pred_global_idx[pi]
        if g_pi in global_matched_pred:
            continue
        for gi, (gx, gy, gr) in enumerate(csv_coords):
            gt_global_key = (idx, gi)
            if gt_global_key in global_matched_gt:
                continue
            dist = np.sqrt((px - gx)**2 + (py - gy)**2)
            if dist < (pr + gr) * 0.5 and abs(pr - gr) < gr:
                global_matched_gt.add(gt_global_key)
                global_matched_pred.add(g_pi)
                break

total_TP = len(global_matched_gt)
total_FP = len(pred_world_dedup) - total_TP

# GT 去重複後總數用 gt_world_dedup，但逐張比對的 GT key 是 (idx, gi)
# 需要算去重複後的 GT 總數
total_GT = len(gt_world_dedup)
total_FN = total_GT - total_TP

total_P  = total_TP / (total_TP + total_FP + 1e-8)
total_R  = total_TP / (total_TP + total_FN + 1e-8)
total_F1 = 2 * total_P * total_R / (total_P + total_R + 1e-8)

print(f"  TP={total_TP}, FP={total_FP}, FN={total_FN}")
print(f"  Precision : {total_P:.4f}")
print(f"  Recall    : {total_R:.4f}")
print(f"  F1        : {total_F1:.4f}")

# =============================================================
# 第二輪：把去重複後的預測轉回各圖像素座標，畫六格圖
# =============================================================
print("\n=== 第二輪：畫圖 ===")

# 按 img_idx 把去重複後的預測分組
from collections import defaultdict
pred_by_img = defaultdict(list)
for (lon, lat, r_deg, img_idx) in pred_world_dedup:
    pred_by_img[img_idx].append((lon, lat, r_deg))

for idx in range(NUM_IMAGES):
    data        = all_data[idx]
    img_disp    = data["img_disp"]
    img_norm    = data["img_norm"]
    pred        = data["pred"]
    csv_coords  = data["csv_coords"]
    longlat_bounds = data["longlat_bounds"]

    h, w = img_disp.shape

    # 去重複後的預測轉回像素座標
    pred_coords_dedup_pix = []
    for (lon, lat, r_deg) in pred_by_img[idx]:
        px, py = longlat_to_pixel(lon, lat, longlat_bounds)
        lon_min, _, lon_max, _ = longlat_bounds
        deg_per_pix = (lon_max - lon_min) / IMG_SIZE
        pr = r_deg / deg_per_pix
        pred_coords_dedup_pix.append((px, py, pr))

    # 逐張 TP/FP/FN（用像素座標算，供圖片標題顯示）
    matched_gt_local   = set()
    matched_pred_local = set()
    for pi, (px, py, pr) in enumerate(pred_coords_dedup_pix):
        for gi, (gx, gy, gr) in enumerate(csv_coords):
            dist = np.sqrt((px - gx)**2 + (py - gy)**2)
            if dist < (pr + gr) * 0.5 and abs(pr - gr) < gr:
                if gi not in matched_gt_local and pi not in matched_pred_local:
                    matched_gt_local.add(gi)
                    matched_pred_local.add(pi)
    TP = len(matched_gt_local)
    FP = len(pred_coords_dedup_pix) - TP
    FN = len(csv_coords) - TP
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    z_surface = make_z_surface(img_norm)

    fig = plt.figure(figsize=(18, 10))

    # [0,0] 原圖
    ax00 = fig.add_subplot(2, 3, 1)
    ax00.imshow(img_disp, cmap="gray")
    ax00.set_title("Original Image")
    ax00.axis("off")

    # [0,1] 模型熱圖
    ax01 = fig.add_subplot(2, 3, 2)
    ax01.imshow(pred, cmap="hot", vmin=0, vmax=1)
    ax01.set_title("Model Heatmap")
    ax01.axis("off")

    # [0,2] Ground Truth
    ax02 = fig.add_subplot(2, 3, 3)
    ax02.imshow(img_disp, cmap="gray")
    for (gx, gy, gr) in csv_coords:
        ax02.add_patch(Circle((gx, gy), gr,
            fill=False, edgecolor="lime", linewidth=1.5))
    ax02.set_title(f"Ground Truth ({len(csv_coords)} craters)")
    ax02.axis("off")

    # [1,0] 模型預測（去重複後）
    ax10 = fig.add_subplot(2, 3, 4)
    ax10.imshow(img_disp, cmap="gray")
    for (px, py, pr) in pred_coords_dedup_pix:
        ax10.add_patch(Circle((px, py), pr,
            fill=False, edgecolor="red", linewidth=1.5))
    ax10.set_title(f"Prediction after dedup ({len(pred_coords_dedup_pix)} craters)")
    ax10.axis("off")

    # [1,1] 結果比對
    ax11 = fig.add_subplot(2, 3, 5)
    ax11.imshow(img_disp, cmap="gray")
    for (gx, gy, gr) in csv_coords:
        ax11.add_patch(Circle((gx, gy), gr,
            fill=False, edgecolor="lime", linewidth=1.5))
    for (px, py, pr) in pred_coords_dedup_pix:
        ax11.add_patch(Circle((px, py), pr,
            fill=False, edgecolor="red", linewidth=1.5))
    ax11.set_title(
        f"GT(green) + Pred(red)\n"
        f"P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}"
    )
    ax11.axis("off")

    # [1,2] 3D
    ax12 = fig.add_subplot(2, 3, 6, projection="3d")
    draw_3d_surface(ax12, z_surface, csv_coords, pred_coords_dedup_pix, h, w)
    ax12.set_title("3D DEM\nGT(green) + Pred(red)")

    plt.suptitle(
        f"[Moon Global Eval] Image {idx:05d} | thresh={TEMPLATE_THRESH} | "
        f"TP={TP} FP={FP} FN={FN}",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"result_{idx:05d}.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    gc.collect()

    if (idx + 1) % 10 == 0:
        print(f"  [{idx+1}/{NUM_IMAGES}] 圖片輸出完成")

# =============================================================
# 最終結果
# =============================================================
print("\n" + "="*55)
print("  全局評估最終結果（去重複後）")
print("="*55)
print(f"  總預測數（去重複後）: {len(pred_world_dedup)}")
print(f"  總 GT 數（去重複後）: {len(gt_world_dedup)}")
print(f"  TP={total_TP}, FP={total_FP}, FN={total_FN}")
print(f"  Precision : {total_P:.4f}")
print(f"  Recall    : {total_R:.4f}")
print(f"  F1        : {total_F1:.4f}")
print(f"\n  圖片輸出：{output_dir}")
