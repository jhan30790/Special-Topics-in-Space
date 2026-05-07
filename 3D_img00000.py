import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource


# =========================
# 路徑設定（自己改）
# =========================
h5_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
gt_csv_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_craters_all.csv"
pred_csv_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\result_img_100_2\arcfit_all_craters_first100.csv"
enhanced_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\dev_images100_enhanced_png"
output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\3d_dem_img"

os.makedirs(output_dir, exist_ok=True)


# =========================
# 基本設定
# =========================
h5_dataset_key = "input_images"
NUM_IMAGES = 100

# 除錯用座標開關
SWAP_XY = False
FLIP_X = False
FLIP_Y = False


# =========================
# 小工具函式
# =========================
def load_dem_from_hdf5(h5_path, image_index, dataset_key="input_images"):
    with h5py.File(h5_path, "r") as f:
        img = f[dataset_key][image_index]

    img = np.array(img)

    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        elif img.shape[0] == 1:
            img = img[0]
        else:
            img = img[:, :, 0]

    return img.astype(np.float32)


def normalize_for_display(img):
    mn, mx = np.min(img), np.max(img)
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def transform_coords(x, y, w, h):
    if SWAP_XY:
        x, y = y, x
    if FLIP_X:
        x = (w - 1) - x
    if FLIP_Y:
        y = (h - 1) - y
    return x, y


def circle_points_3d(z_surface, xc, yc, r, n_points=200, z_offset=0.03):
    h, w = z_surface.shape
    theta = np.linspace(0, 2 * np.pi, n_points)

    xs = xc + r * np.cos(theta)
    ys = yc + r * np.sin(theta)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[valid]
    ys = ys[valid]

    if len(xs) == 0:
        return np.array([]), np.array([]), np.array([])

    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)

    zs = z_surface[yi, xi] + z_offset

    return xs, ys, zs


def match_craters(gt_df, pred_df):
    candidates = []

    for gi, gt in gt_df.iterrows():
        xg, yg = gt["x_plot"], gt["y_plot"]
        rg = gt["r_pix"]

        for pi, pr in pred_df.iterrows():
            xp, yp = pr["x_plot"], pr["y_plot"]
            rp = pr["radius"]

            center_dist = np.hypot(xg - xp, yg - yp)
            radius_diff = abs(rg - rp)
            tol = 0.5 * max(rg, rp)

            if center_dist <= tol and radius_diff <= tol:
                score = center_dist + radius_diff
                candidates.append((score, gi, pi))

    candidates.sort(key=lambda x: x[0])

    gt_used = set()
    pred_used = set()
    matches = []

    for score, gi, pi in candidates:
        if gi not in gt_used and pi not in pred_used:
            gt_used.add(gi)
            pred_used.add(pi)
            matches.append((gi, pi, score))

    return matches, gt_used, pred_used


def make_z_surface(dem):
    dem_smooth = gaussian_filter(dem, sigma=1.5)

    p1, p99 = np.percentile(dem_smooth, (1, 99))
    dem_clip = np.clip(dem_smooth, p1, p99)

    z_surface = (dem_clip - p1) / (p99 - p1 + 1e-8)

    return z_surface


def add_plot_coords(gt_df, pred_df, w, h):
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()

    gt_df["r_pix"] = gt_df["Diameter (pix)"] / 2.0

    gt_x_plot, gt_y_plot = [], []
    for _, row in gt_df.iterrows():
        x2, y2 = transform_coords(row["x"], row["y"], w, h)
        gt_x_plot.append(x2)
        gt_y_plot.append(y2)

    gt_df["x_plot"] = gt_x_plot
    gt_df["y_plot"] = gt_y_plot

    pred_x_plot, pred_y_plot = [], []
    for _, row in pred_df.iterrows():
        x2, y2 = transform_coords(row["x_center"], row["y_center"], w, h)
        pred_x_plot.append(x2)
        pred_y_plot.append(y2)

    pred_df["x_plot"] = pred_x_plot
    pred_df["y_plot"] = pred_y_plot

    return gt_df, pred_df


# =========================
# 讀 CSV
# =========================
gt_all = pd.read_csv(gt_csv_path)
pred_all = pd.read_csv(pred_csv_path)


# =========================
# 批次畫圖
# =========================
for image_index_to_plot in range(NUM_IMAGES):
    print(f"Processing image_index = {image_index_to_plot}")

    # 讀 DEM
    dem = load_dem_from_hdf5(
        h5_path,
        image_index_to_plot,
        dataset_key=h5_dataset_key
    )

    h, w = dem.shape
    Y, X = np.mgrid[0:h, 0:w]

    # 讀強化影像
    enhanced_path = os.path.join(
        enhanced_dir,
        f"dev_image_{image_index_to_plot:05d}.png"
    )

    if os.path.exists(enhanced_path):
        img_enhanced = np.array(Image.open(enhanced_path))
    else:
        print(f"Warning: 找不到強化影像，改用 DEM 顯示：{enhanced_path}")
        img_enhanced = normalize_for_display(dem)

    # 建 3D 高度
    z_surface = make_z_surface(dem)

    # 只抓同一張圖的 GT / Prediction
    gt_df = gt_all[gt_all["image_index"] == image_index_to_plot].copy()
    pred_df = pred_all[pred_all["image_index"] == image_index_to_plot].copy()

    # 如果這張沒有資料，也可以繼續畫底圖
    gt_df, pred_df = add_plot_coords(gt_df, pred_df, w, h)

    # matching
    matches, matched_gt_idx, matched_pred_idx = match_craters(gt_df, pred_df)

    print(f"Image index = {image_index_to_plot}")
    print(f"GT count    = {len(gt_df)}")
    print(f"Pred count  = {len(pred_df)}")
    print(f"Matched     = {len(matches)}")

    # =========================
    # 畫圖
    # =========================
    fig = plt.figure(figsize=(16, 7))

    # -------- 2D --------
    ax1 = fig.add_subplot(1, 2, 1)

    if img_enhanced.ndim == 2:
        ax1.imshow(img_enhanced, cmap="gray", origin="upper")
    else:
        ax1.imshow(img_enhanced, origin="upper")

    ax1.set_title(f"2D Enhanced DEM Overlay (image_index={image_index_to_plot})")
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)

    # GT in 2D
    for gi, row in gt_df.iterrows():
        edge_color = "lime" if gi in matched_gt_idx else "green"

        circ = Circle(
            (row["x_plot"], row["y_plot"]),
            row["r_pix"],
            fill=False,
            edgecolor="green",
            linewidth=2.0
        )
        ax1.add_patch(circ)

    # Prediction in 2D
    for pi, row in pred_df.iterrows():
        edge_color = "lime" if pi in matched_pred_idx else "red"

        circ = Circle(
            (row["x_plot"], row["y_plot"]),
            row["radius"],
            fill=False,
            edgecolor="red",
            linewidth=1.8
        )
        ax1.add_patch(circ)

    # -------- 3D --------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(
        z_surface,
        cmap=plt.cm.terrain,
        vert_exag=1,
        blend_mode="soft"
    )

    ax2.plot_surface(
        X, Y, z_surface,
        facecolors=rgb,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.85
    )

    # GT in 3D
    for _, row in gt_df.iterrows():
        x = row["x_plot"]
        y = row["y_plot"]
        r = row["r_pix"]

        # 跳過碰到邊界的圓，避免 3D 出現怪線
        if (x - r < 0) or (x + r >= w) or (y - r < 0) or (y + r >= h):
            continue

        xs, ys, zs = circle_points_3d(
            z_surface,
            x,
            y,
            r,
            n_points=200,
            z_offset=0.05
        )

        if len(xs) > 0:
            ax2.plot(xs, ys, zs, color="deepskyblue", linewidth=2)

    # Prediction in 3D
    for _, row in pred_df.iterrows():
        x = row["x_plot"]
        y = row["y_plot"]
        r = row["radius"]

        # 跳過碰到邊界的圓，避免 3D 出現怪線
        if (x - r < 0) or (x + r >= w) or (y - r < 0) or (y + r >= h):
            continue

        xs, ys, zs = circle_points_3d(
            z_surface,
            x,
            y,
            r,
            n_points=200,
            z_offset=0.03
        )

        if len(xs) > 0:
            ax2.plot(xs, ys, zs, color="red", linewidth=2)

    ax2.set_title(f"3D DEM Overlay (image_index={image_index_to_plot})")

    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_zlim(0, np.max(z_surface) + 0.1)

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Height")

    ax2.view_init(elev=35, azim=-60)

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"dem_overlay_3d_{image_index_to_plot:05d}.png"
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved to:", save_path)

print("全部完成")
