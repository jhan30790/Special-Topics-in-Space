import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PIL import Image
import numpy as np

from scipy.ndimage import gaussian_filter

# =========================
# 路徑設定（自己改）
# =========================
h5_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
gt_csv_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_craters_all.csv"
pred_csv_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\result_img_100_2\arcfit_all_craters_first100.csv"
output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\3d_dem_img"
os.makedirs(output_dir, exist_ok=True)

img_enhanced = np.array(Image.open(r"C:\Users\jhan3\OneDrive\桌面\大二專題\dev_images100_enhanced_png\dev_image_00000.png"))

# 要視覺化哪一張
image_index_to_plot = 0

# hdf5 裡面的 dataset key
h5_dataset_key = "input_images"

# =========================
# 除錯用座標開關
# 如果你發現座標明顯錯位，可以試這些
# =========================
SWAP_XY = False
FLIP_X = False
FLIP_Y = False

# 3D 顯示用高度誇張倍率
VERTICAL_EXAGGERATION = 10.0


# =========================
# 小工具函式
# =========================
def load_dem_from_hdf5(h5_path, image_index, dataset_key="input_images"):
    with h5py.File(h5_path, "r") as f:
        print("HDF5 keys:", list(f.keys()))
        img = f[dataset_key][image_index]

    img = np.array(img)

    # 可能是 (256,256,1) 或 (1,256,256) 或 (256,256)
    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        elif img.shape[0] == 1:
            img = img[0]
        else:
            # 如果不是單通道，先取第一個通道
            img = img[:, :, 0]

    return img.astype(np.float32)


def normalize_for_display(img):
    mn, mx = np.min(img), np.max(img)
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def make_3d_height(img, exaggeration=50.0):
    """把 DEM 拉成比較容易看的高度"""
    z = normalize_for_display(img) * exaggeration
    return z


def transform_coords(x, y, w, h):
    if SWAP_XY:
        x, y = y, x
    if FLIP_X:
        x = (w - 1) - x
    if FLIP_Y:
        y = (h - 1) - y
    return x, y


def circle_points_3d(z_surface, xc, yc, r, n_points=200, z_offset=1.0):
    """把 2D circle 轉成 3D 曲面上的圓"""
    h, w = z_surface.shape
    theta = np.linspace(0, 2*np.pi, n_points)

    xs = xc + r * np.cos(theta)
    ys = yc + r * np.sin(theta)

    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)

    zs = z_surface[yi, xi] + z_offset
    return xs, ys, zs


def match_craters(gt_df, pred_df):
    """
    簡單 matching：
    - center distance <= 0.5 * max(r_gt, r_pred)
    - radius difference <= 0.5 * max(r_gt, r_pred)
    用 greedy matching
    """
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

def circle_points_3d(z_surface, xc, yc, r, n_points=200, z_offset=1.0):
    import numpy as np

    h, w = z_surface.shape
    theta = np.linspace(0, 2*np.pi, n_points)

    xs = xc + r * np.cos(theta)
    ys = yc + r * np.sin(theta)

    # 先保留在範圍內的點
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[valid]
    ys = ys[valid]

    # 關鍵修正：轉成整數後再 clip 一次，避免 round 到 256
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)

    zs = z_surface[yi, xi] + z_offset

    return xs, ys, zs


# =========================
# 讀資料
# =========================
dem = load_dem_from_hdf5(h5_path, image_index_to_plot, dataset_key=h5_dataset_key)
dem_show = normalize_for_display(dem)
# 讓 3D 地形凹凸更明顯
p2, p98 = np.percentile(dem, (2, 98))
dem_clip = np.clip(dem, p2, p98)

dem_smooth = gaussian_filter(dem, sigma=1.5)

p1, p99 = np.percentile(dem_smooth, (1, 99))
dem_clip = np.clip(dem_smooth, p1, p99)

z_surface = (dem_clip - p1) / (p99 - p1 + 1e-8)

h, w = dem.shape
Y, X = np.mgrid[0:h, 0:w]

gt_all = pd.read_csv(gt_csv_path)
pred_all = pd.read_csv(pred_csv_path)

# 只抓同一張圖
gt_df = gt_all[gt_all["image_index"] == image_index_to_plot].copy()
pred_df = pred_all[pred_all["image_index"] == image_index_to_plot].copy()

# ground truth 半徑
gt_df["r_pix"] = gt_df["Diameter (pix)"] / 2.0

# 座標轉換（如果有需要翻轉/交換）
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
ax1.imshow(img_enhanced, cmap="gray", origin="upper")
ax1.set_title(f"2D DEM Overlay (image_index={image_index_to_plot})")
ax1.set_xlim(0, w)
ax1.set_ylim(h, 0)

# 畫 GT
for gi, row in gt_df.iterrows():
    color = "lime" if gi in matched_gt_idx else "deepskyblue"
    circ = Circle(
        (row["x_plot"], row["y_plot"]),
        row["r_pix"],
        fill=False,
        edgecolor="green",
        linewidth=2.0,
    )
    ax1.add_patch(circ)

# 畫 Prediction
for pi, row in pred_df.iterrows():
    color = "lime" if pi in matched_pred_idx else "red"
    circ = Circle(
        (row["x_plot"], row["y_plot"]),
        row["radius"],
        fill=False,
        edgecolor="red",
        linewidth=1.8
    )
    ax1.add_patch(circ)

# ax1.text(5, 15, "Blue dashed = GT\nRed = Prediction\nGreen = Matched",
         # color="yellow", fontsize=10,
         # bbox=dict(facecolor="black", alpha=0.5))

# -------- 3D --------
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

from matplotlib.colors import LightSource

ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(z_surface, cmap=plt.cm.terrain, vert_exag=1, blend_mode='soft')

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
    xs, ys, zs = circle_points_3d(
        z_surface,
        row["x_plot"],
        row["y_plot"],
        row["r_pix"],
        n_points=200,
        z_offset=0.05
    )
    ax2.plot(xs, ys, zs, color="deepskyblue", linewidth=2)

# Prediction in 3D
for _, row in pred_df.iterrows():
    xs, ys, zs = circle_points_3d(
        z_surface,
        row["x_plot"],
        row["y_plot"],
        row["radius"],
        n_points=200,
        z_offset=0.03
    )
    ax2.plot(xs, ys, zs, color="red", linewidth=2)

ax2.set_title("3D DEM Overlay")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Height")
ax2.view_init(elev=45, azim=-60)
# ax2.invert_yaxis()

ax2.set_title(f"3D DEM Overlay (image_index={image_index_to_plot})")

ax2.set_xlim(0, w)
ax2.set_ylim(h, 0)
ax2.set_zlim(0, np.max(z_surface) + 0.1)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Height")

ax2.view_init(elev=35, azim=-60)

plt.tight_layout()

save_path = os.path.join(output_dir, f"dem_overlay_3d_{image_index_to_plot:05d}.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved to:", save_path)
