import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

from matplotlib.patches import Circle

# =============================================================
# DeepMoon repo 路徑
# =============================================================
DEEPMOON_DIR = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master"
sys.path.insert(0, DEEPMOON_DIR)

import utils.template_match_target as tmt

# =============================================================
# 路徑設定
# =============================================================
model_path        = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"
mars_images_path  = r"C:\Users\jhan3\OneDrive\桌面\大二專題\deepmars-master\data\processed\mars_1000_images_00000.hdf5"
mars_craters_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\deepmars-master\data\processed\mars_1000_craters_00000.hdf5"

output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\result_mars_zeroshot"
os.makedirs(output_dir, exist_ok=True)

# =============================================================
# 固定參數（跟月球版完全一樣）
# =============================================================
MINRAD          = 5
MAXRAD          = 40
TARGET_THRESH   = 0.1
LONGLAT_THRESH2 = 1.8
RAD_THRESH      = 1.0

# 掃描這些 threshold（從低開始，因為 zero-shot 信心值會偏低）
TEMPLATE_THRESH_LIST = [ 0.35, 0.40, 0.45,0.5]

# =============================================================
# 讀取模型
# =============================================================
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded.\n")

# =============================================================
# 讀取火星圖片並跑推論
# =============================================================
all_preds      = []
all_csv_coords = []
all_img_disp   = []
img_ids        = []

print("Running model predictions on Mars data...")
with h5py.File(mars_images_path,  "r") as f_img, \
     h5py.File(mars_craters_path, "r") as f_crt:

    images = f_img["input_images"][:]
    sorted_ids = sorted(f_crt.keys())

    for i, img_id in enumerate(sorted_ids):
        if i >= len(images):
            break

        # 前處理（跟月球版一樣）
        img = images[i].astype("float32")
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
        try:
            cols  = [x.decode() for x in f_crt[img_id]['axis0'][:]]
            vals  = f_crt[img_id]['block0_values'][:]
            df    = dict(zip(cols, vals.T))
            x_pix = df['x']
            y_pix = df['y']
            r_pix = df['Diameter (pix)'] / 2.0
            csv_coords = np.column_stack([x_pix, y_pix, r_pix])
        except Exception:
            csv_coords = np.empty((0, 3))

        # 顯示用圖
        p2, p98 = np.percentile(img_norm, (2, 98))
        img_disp = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

        all_preds.append(pred)
        all_csv_coords.append(csv_coords)
        all_img_disp.append(img_disp)
        img_ids.append(img_id)

        print(f"  [{img_id}] GT craters: {len(csv_coords)}")

print(f"\nDone. {len(all_preds)} images predicted.\n")

# =============================================================
# 對每個 threshold 跑評估
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

    for idx, img_id in enumerate(img_ids):
        pred       = all_preds[idx]
        csv_coords = all_csv_coords[idx]
        img_disp   = all_img_disp[idx]

        if len(csv_coords) > 0:
            (N_match, N_csv, N_detect, maxr,
             err_lo, err_la, err_r, frac_dupes) = tmt.template_match_t2c(
                pred.copy(),
                csv_coords.copy(),
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

        # 畫圖
        pred_coords = tmt.template_match_t(
            pred.copy(),
            minrad=MINRAD, maxrad=MAXRAD,
            longlat_thresh2=LONGLAT_THRESH2,
            rad_thresh=RAD_THRESH,
            template_thresh=thresh,
            target_thresh=TARGET_THRESH
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(img_disp, cmap="gray")
        for (gx, gy, gr) in csv_coords:
            ax[0].add_patch(Circle((gx, gy), gr,
                fill=False, edgecolor="lime", linewidth=1.5))
        ax[0].set_title(f"GT ({len(csv_coords)} craters)")
        ax[0].axis("off")

        ax[1].imshow(img_disp, cmap="gray")
        for (gx, gy, gr) in csv_coords:
            ax[1].add_patch(Circle((gx, gy), gr,
                fill=False, edgecolor="lime", linewidth=1.5))
        for (px, py, pr) in pred_coords:
            ax[1].add_patch(Circle((px, py), pr,
                fill=False, edgecolor="red", linewidth=1.5))
        ax[1].set_title(
            f"GT(green) + Pred(red)\n"
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
        )
        ax[1].axis("off")

        plt.suptitle(f"[Mars] {img_id} | thresh={thresh}", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(thresh_dir, f"result_{img_id}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        gc.collect()

    # 總結這個 threshold
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
    })

# =============================================================
# 最終比較表
# =============================================================
print("\n" + "="*65)
print("  火星 Zero-Shot Threshold Sweep 結果")
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

print("\n  對比月球最佳結果（threshold=0.35）：")
print(f"  月球 Precision: 0.7631  |  火星最佳: {best['Precision']:.4f}")
print(f"  月球 Recall:    0.2361  |  火星最佳: {best['Recall']:.4f}")
print(f"  月球 F1:        0.3606  |  火星最佳: {best['F1']:.4f}")
print(f"\n  圖片輸出：{output_dir}")
