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
# DeepMoon repo 路徑
# =============================================================
DEEPMOON_DIR = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master"
sys.path.insert(0, DEEPMOON_DIR)

import utils.template_match_target as tmt

# =============================================================
# 路徑設定
# =============================================================
dev_images_path  = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
dev_craters_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_craters.hdf5"
model_path       = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"

output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\result_deepmoon_eval"
os.makedirs(output_dir, exist_ok=True)

# =============================================================
# 執行張數
# =============================================================
NUM_IMAGES = 100

# =============================================================
# 固定參數
# =============================================================
MINRAD          = 5
MAXRAD          = 40
TARGET_THRESH   = 0.1
LONGLAT_THRESH2 = 1.8
RAD_THRESH      = 1.0

TEMPLATE_THRESH_LIST = [0.35,0.45,0.5]

# =============================================================
# 3D 工具函式
# =============================================================
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
    # GT 綠色
    for (gx, gy, gr) in craters_gt:
        if (gx - gr < 0) or (gx + gr >= w) or (gy - gr < 0) or (gy + gr >= h):
            continue
        xs, ys, zs = circle_points_3d(z_surface, gx, gy, gr, z_offset=0.05)
        if len(xs) > 0:
            ax.plot(xs, ys, zs, color="lime", linewidth=2)
    # Pred 紅色
    for (px, py, pr) in craters_pred:
        if (px - pr < 0) or (px + pr >= w) or (py - pr < 0) or (py + pr >= h):
            continue
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
# 讀取模型
# =============================================================
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded.\n")

# =============================================================
# 讀取所有圖片的預測結果（只跑一次模型）
# =============================================================
all_preds      = []
all_csv_coords = []
all_img_disp   = []
all_img_norm   = []

print("Running model predictions...")
with h5py.File(dev_images_path,  "r") as f_img, \
     h5py.File(dev_craters_path, "r") as f_crt:

    total_imgs = f_img["input_images"].shape[0]
    run_count  = min(NUM_IMAGES, total_imgs)

    for idx in range(run_count):
        img = f_img["input_images"][idx].astype("float32")
        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)

        x = img_norm.reshape(1, 256, 256, 1)
        pred_raw = model.predict(x, verbose=0)

        if pred_raw.ndim == 4:
            pred = pred_raw[0, :, :, 0]
        elif pred_raw.ndim == 3:
            pred = pred_raw[0, :, :]
        else:
            raise ValueError(f"Unexpected shape: {pred_raw.shape}")

        # 讀 GT
        id_str = f"img_{idx:05d}"
        try:
            block = f_crt[id_str]['block0_values'][...]
            if block.ndim == 1:
                block = block.reshape(1, -1)
            if block.shape[0] == 6 and block.shape[1] != 6:
                block = block.T
            if block.shape[1] < 6:
                csv_coords = np.empty((0, 3))
            else:
                x_pix = block[:, 3]
                y_pix = block[:, 4]
                r_pix = block[:, 5] / 2.0
                csv_coords = np.column_stack([x_pix, y_pix, r_pix])
        except KeyError:
            csv_coords = np.empty((0, 3))

        p2, p98 = np.percentile(img_norm, (2, 98))
        img_disp = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

        all_preds.append(pred)
        all_csv_coords.append(csv_coords)
        all_img_disp.append(img_disp)
        all_img_norm.append(img_norm)

        if (idx + 1) % 10 == 0:
            print(f"  Predicted {idx+1}/{run_count}...")

print(f"Done. {run_count} images predicted.\n")

# =============================================================
# 對每個 threshold 跑評估 + 畫 2x3 六格圖
# =============================================================
sweep_results = []

for thresh in TEMPLATE_THRESH_LIST:
    print(f"{'='*55}")
    print(f"  template_thresh = {thresh}")
    print(f"{'='*55}")

    thresh_dir = os.path.join(output_dir, f"thresh_{thresh}")
    os.makedirs(thresh_dir, exist_ok=True)

    total_TP = 0
    total_FP = 0
    total_FN = 0
    err_lo_all = []
    err_la_all = []
    err_r_all  = []

    for idx in range(run_count):
        pred       = all_preds[idx]
        csv_coords = all_csv_coords[idx]
        img_disp   = all_img_disp[idx]
        img_norm   = all_img_norm[idx]

        h, w = img_disp.shape

        # 評估
        if len(csv_coords) > 0:
            (N_match, N_csv, N_detect, maxr,
             err_lo, err_la, err_r, frac_dupes) = tmt.template_match_t2c(
                pred.copy(), csv_coords.copy(),
                minrad=MINRAD, maxrad=MAXRAD,
                longlat_thresh2=LONGLAT_THRESH2,
                rad_thresh=RAD_THRESH,
                template_thresh=thresh,
                target_thresh=TARGET_THRESH
            )
        else:
            templ_coords = tmt.template_match_t(
                pred.copy(),
                minrad=MINRAD, maxrad=MAXRAD,
                longlat_thresh2=LONGLAT_THRESH2,
                rad_thresh=RAD_THRESH,
                template_thresh=thresh,
                target_thresh=TARGET_THRESH
            )
            N_match, N_csv, N_detect = 0, 0, len(templ_coords)
            err_lo = err_la = err_r = 0.0

        TP = N_match
        FP = max(N_detect - N_match, 0)
        FN = max(N_csv - N_match, 0)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        if N_match >= 1:
            err_lo_all.append(err_lo)
            err_la_all.append(err_la)
            err_r_all.append(err_r)

        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        # 畫圖（前 20 張）
        if idx < 100:
            pred_coords = tmt.template_match_t(
                pred.copy(),
                minrad=MINRAD, maxrad=MAXRAD,
                longlat_thresh2=LONGLAT_THRESH2,
                rad_thresh=RAD_THRESH,
                template_thresh=thresh,
                target_thresh=TARGET_THRESH
            )

            z_surface = make_z_surface(img_norm)

            # 2x3 六格圖
            fig = plt.figure(figsize=(18, 10))

            # 上排
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

            # [0,2] Ground Truth（綠）
            ax02 = fig.add_subplot(2, 3, 3)
            ax02.imshow(img_disp, cmap="gray")
            for (gx, gy, gr) in csv_coords:
                ax02.add_patch(Circle((gx, gy), gr,
                    fill=False, edgecolor="lime", linewidth=1.5))
            ax02.set_title(f"Ground Truth ({len(csv_coords)} craters)")
            ax02.axis("off")

            # 下排
            # [1,0] 預測（紅）
            ax10 = fig.add_subplot(2, 3, 4)
            ax10.imshow(img_disp, cmap="gray")
            for (px, py, pr) in pred_coords:
                ax10.add_patch(Circle((px, py), pr,
                    fill=False, edgecolor="red", linewidth=1.5))
            ax10.set_title(f"Prediction ({len(pred_coords)} craters)")
            ax10.axis("off")

            # [1,1] GT + 預測疊加
            ax11 = fig.add_subplot(2, 3, 5)
            ax11.imshow(img_disp, cmap="gray")
            for (gx, gy, gr) in csv_coords:
                ax11.add_patch(Circle((gx, gy), gr,
                    fill=False, edgecolor="lime", linewidth=1.5))
            for (px, py, pr) in pred_coords:
                ax11.add_patch(Circle((px, py), pr,
                    fill=False, edgecolor="red", linewidth=1.5))
            ax11.set_title(
                f"GT(green) + Pred(red)\n"
                f"P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}"
            )
            ax11.axis("off")

            # [1,2] 3D 疊加
            ax12 = fig.add_subplot(2, 3, 6, projection="3d")
            draw_3d_surface(ax12, z_surface, csv_coords, pred_coords, h, w)
            ax12.set_title("3D DEM\nGT(green) + Pred(red)")

            plt.suptitle(
                f"[Moon] Image {idx:05d} | thresh={thresh} | "
                f"TP={TP} FP={FP} FN={FN}",
                fontsize=13
            )
            plt.tight_layout()
            fig.savefig(os.path.join(thresh_dir, f"result_{idx:05d}.png"),
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
            gc.collect()

    # 總結
    total_precision = total_TP / (total_TP + total_FP + 1e-8)
    total_recall    = total_TP / (total_TP + total_FN + 1e-8)
    total_f1        = (2 * total_precision * total_recall
                       / (total_precision + total_recall + 1e-8))
    mean_err_lo = np.mean(err_lo_all) if err_lo_all else 0.0
    mean_err_la = np.mean(err_la_all) if err_la_all else 0.0
    mean_err_r  = np.mean(err_r_all)  if err_r_all  else 0.0

    print(f"  TP={total_TP}, FP={total_FP}, FN={total_FN}")
    print(f"  Precision : {total_precision:.4f}")
    print(f"  Recall    : {total_recall:.4f}")
    print(f"  F1-score  : {total_f1:.4f}")
    print(f"  Long err  : {mean_err_lo*100:.1f}%")
    print(f"  Lat err   : {mean_err_la*100:.1f}%")
    print(f"  Rad err   : {mean_err_r*100:.1f}%\n")

    sweep_results.append({
        "template_thresh": thresh,
        "Precision": total_precision,
        "Recall":    total_recall,
        "F1":        total_f1,
        "Long_err":  mean_err_lo * 100,
        "Lat_err":   mean_err_la * 100,
        "Rad_err":   mean_err_r  * 100,
    })

# =============================================================
# 最終比較表
# =============================================================
print("\n" + "="*65)
print("  Threshold Sweep 結果比較")
print("="*65)
print(f"  {'Thresh':>8} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("-"*65)
best = max(sweep_results, key=lambda x: x["F1"])
for r in sweep_results:
    marker = " <- best F1" if r["template_thresh"] == best["template_thresh"] else ""
    print(f"  {r['template_thresh']:>8} | "
          f"{r['Precision']:>10.4f} | "
          f"{r['Recall']:>8.4f} | "
          f"{r['F1']:>8.4f}{marker}")
print("="*65)
print(f"\n  最佳 threshold = {best['template_thresh']}")
print(f"  Best F1        = {best['F1']:.4f}")
print(f"  Precision      = {best['Precision']:.4f}")
print(f"  Recall         = {best['Recall']:.4f}")

print("\n  [論文 Post-Processed Test 參考]")
print("  Recall=92%, Precision=56%")
print(f"\n  圖片輸出：{output_dir}")
