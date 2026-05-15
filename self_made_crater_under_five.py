import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
from matplotlib.patches import Circle
from itertools import product

# =============================================================
# 路徑設定
# =============================================================
DEEPMOON_DIR = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master"
sys.path.insert(0, DEEPMOON_DIR)
import utils.template_match_target as tmt

model_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"
output_dir = r"C:\Users\jhan3\OneDrive\桌面\大二專題\selfmade_image_test"
os.makedirs(output_dir, exist_ok=True)

# =============================================================
# 固定參數
# =============================================================
IMG_SIZE        = 256
MINRAD          = 5
MAXRAD          = 40
TEMPLATE_THRESH = 0.35
TARGET_THRESH   = 0.1
LONGLAT_THRESH2 = 1.8
RAD_THRESH      = 1.0

# =============================================================
# 載入模型
# =============================================================
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded.\n")

# =============================================================
# 產生底圖函數
# =============================================================
def make_background(size=256, base=0.5, noise_std=0.02, seed=42):
    """均勻灰階底圖 + 輕微噪聲"""
    rng = np.random.default_rng(seed)
    bg = np.full((size, size), base, dtype=np.float32)
    bg += rng.normal(0, noise_std, (size, size)).astype(np.float32)
    return np.clip(bg, 0, 1)

# =============================================================
# 產生單一坑洞函數
# =============================================================
def add_crater(img, cx, cy, radius, depth):
    """
    在 img 上疊加一個碗狀隕石坑
    - 坑底：往下凹 (減 depth)
    - 坑緣：輕微隆起 (加 depth*0.3)
    """
    size = img.shape[0]
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float32)

    # 坑底凹陷（高斯碗）
    sigma_bowl = radius * 0.6
    bowl = -depth * np.exp(-dist**2 / (2 * sigma_bowl**2))

    # 坑緣隆起（環形高斯）
    sigma_rim = radius * 0.3
    rim = depth * 0.3 * np.exp(-(dist - radius)**2 / (2 * sigma_rim**2))

    img += bowl + rim
    return img

# =============================================================
# 產生圖像函數（最多 5 個坑，坑間強制最小間距）
# =============================================================
def generate_image(n_craters, radius, depth, size=256, seed=0, min_sep_factor=2.5):
    """
    產生一張含最多 n_craters 個假坑的圖（上限 5 個），回傳圖和坑的座標列表。
    坑洞之間的中心距離至少為 radius * min_sep_factor，確保不會黏在一起。

    參數：
        n_craters     : 目標坑洞數量（最多 5 個）
        radius        : 坑洞半徑（像素）
        depth         : 坑洞深度（影響亮暗對比）
        size          : 圖像邊長（預設 256）
        seed          : 隨機種子（確保可重現）
        min_sep_factor: 最小中心間距 = radius * min_sep_factor
    """
    # 強制上限 5 個
    n_craters = min(n_craters, 5)

    rng = np.random.default_rng(seed)
    img = make_background(size=size, seed=seed)

    coords = []           # 已放置坑洞的 (cx, cy, radius)
    margin = radius + 5   # 距邊界至少 margin 像素
    min_dist = radius * min_sep_factor  # 兩坑中心最小距離
    max_attempts = 2000   # 防止無限迴圈
    attempts = 0

    while len(coords) < n_craters and attempts < max_attempts:
        cx = int(rng.integers(margin, size - margin))
        cy = int(rng.integers(margin, size - margin))

        # 檢查與所有已放置坑洞的距離
        too_close = False
        for (ex, ey, er) in coords:
            dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if dist < min_dist:
                too_close = True
                break

        if not too_close:
            img = add_crater(img, cx, cy, radius, depth)
            coords.append((cx, cy, radius))

        attempts += 1

    img = np.clip(img, 0, 1)
    return img, coords

# =============================================================
# 實驗網格（density 上限設為 5）
# =============================================================
radius_cases = [
    ("small",  7),
    ("medium", 17),
    ("large",  32),
]
depth_cases = [
    ("shallow", 0.1),
    ("medium",  0.3),
    ("deep",    0.5),
]
density_cases = [
    ("sparse",   2),
    ("moderate", 3),
    ("dense",    5),   # 最多 5 個
]

# =============================================================
# 主實驗迴圈
# =============================================================
results = []
total = len(radius_cases) * len(depth_cases) * len(density_cases)
count = 0

for (r_name, radius), (d_name, depth), (den_name, n_craters) in product(
        radius_cases, depth_cases, density_cases):

    count += 1
    label = f"{r_name}_depth{d_name}_den{den_name}"
    print(f"[{count}/{total}] {label}")

    # 產生圖（坑洞最多 5 個，且彼此間距充足）
    img, gt_coords = generate_image(
        n_craters, radius, depth, seed=count, min_sep_factor=2.5
    )
    gt_coords = np.array(gt_coords, dtype=np.float32)  # shape (N, 3)

    # 前處理
    img_min, img_max = img.min(), img.max()
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)
    x = img_norm.reshape(1, 256, 256, 1)

    # 模型推論
    pred_raw = model.predict(x, verbose=0)
    if pred_raw.ndim == 4:
        pred = pred_raw[0, :, :, 0]
    elif pred_raw.ndim == 3:
        pred = pred_raw[0, :, :]

    # 後處理：模板匹配找坑洞
    pred_coords = tmt.template_match_t(
        pred.copy(),
        minrad=MINRAD, maxrad=MAXRAD,
        longlat_thresh2=LONGLAT_THRESH2,
        rad_thresh=RAD_THRESH,
        template_thresh=TEMPLATE_THRESH,
        target_thresh=TARGET_THRESH
    )

    # 評估：IoU-based matching（中心距離 + 半徑差）
    matched_gt   = set()
    matched_pred = set()
    for pi, (px, py, pr) in enumerate(pred_coords):
        for gi, (gx, gy, gr) in enumerate(gt_coords):
            dist = np.sqrt((px - gx)**2 + (py - gy)**2)
            if dist < (pr + gr) * 0.5 and abs(pr - gr) < gr:
                if gi not in matched_gt and pi not in matched_pred:
                    matched_gt.add(gi)
                    matched_pred.add(pi)

    TP = len(matched_gt)
    FP = len(pred_coords) - TP
    FN = len(gt_coords) - TP
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    results.append({
        "label":     label,
        "radius":    r_name,
        "depth":     d_name,
        "density":   den_name,
        "n_gt":      len(gt_coords),
        "n_pred":    len(pred_coords),
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
    })

    # 顯示圖：輸入 / 熱圖 / 標注結果
    p2, p98 = np.percentile(img_norm, (2, 98))
    img_disp = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="hot")
    axes[1].set_title("Model Output (heatmap)")
    axes[1].axis("off")

    axes[2].imshow(img_disp, cmap="gray")
    for (gx, gy, gr) in gt_coords:
        axes[2].add_patch(Circle((gx, gy), gr,
            fill=False, edgecolor="lime", linewidth=1.5))
    for (px, py, pr) in pred_coords:
        axes[2].add_patch(Circle((px, py), pr,
            fill=False, edgecolor="red", linewidth=1.5))
    axes[2].set_title(
        f"GT(green) Pred(red)\nP={precision:.2f} R={recall:.2f} F1={f1:.2f}"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"{label} | r={radius}px  depth={depth}  n_gt={len(gt_coords)}", fontsize=11
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{label}.png"), dpi=100, bbox_inches="tight"
    )
    plt.close(fig)
    gc.collect()

# =============================================================
# 印出結果總表
# =============================================================
print("\n" + "="*80)
print("  壓力測試結果總表")
print("="*80)
print(f"  {'條件':<40} | {'GT':>4} | {'Pred':>4} | {'P':>6} | {'R':>6} | {'F1':>6}")
print("-"*80)
for r in results:
    print(f"  {r['label']:<40} | {r['n_gt']:>4} | {r['n_pred']:>4} | "
          f"{r['Precision']:>6.3f} | {r['Recall']:>6.3f} | {r['F1']:>6.3f}")
print("="*80)
print(f"\n圖片輸出：{output_dir}")

# =============================================================
# Shallow 條件 Threshold Sweep
# =============================================================
print("\n" + "="*80)
print("  Shallow 深度 Threshold Sweep")
print("="*80)

THRESH_LIST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

shallow_cases = [
    ("small",  7),
    ("medium", 17),
    ("large",  32),
]
density_sweep = [
    ("sparse",   2),
    ("moderate", 3),
    ("dense",    5),
]

shallow_dir = os.path.join(output_dir, "shallow_sweep")
os.makedirs(shallow_dir, exist_ok=True)

# 先把所有 shallow 條件的圖和 GT 準備好（固定 seed 跟主實驗一致）
shallow_images = {}
for (r_name, radius), (den_name, n_craters) in product(shallow_cases, density_sweep):
    r_idx   = ["small", "medium", "large"].index(r_name)
    den_idx = ["sparse", "moderate", "dense"].index(den_name)
    seed = r_idx * 9 + 0 * 3 + den_idx + 1

    img, gt_coords = generate_image(
        n_craters, radius, depth=0.1, seed=seed, min_sep_factor=2.5
    )
    gt_coords = np.array(gt_coords, dtype=np.float32)

    img_min, img_max = img.min(), img.max()
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)
    x = img_norm.reshape(1, 256, 256, 1)

    pred_raw = model.predict(x, verbose=0)
    if pred_raw.ndim == 4:
        pred = pred_raw[0, :, :, 0]
    elif pred_raw.ndim == 3:
        pred = pred_raw[0, :, :]

    p2, p98 = np.percentile(img_norm, (2, 98))
    img_disp = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

    shallow_images[(r_name, den_name)] = {
        "pred":     pred,
        "gt":       gt_coords,
        "img_disp": img_disp,
        "radius":   radius,
        "n_craters": n_craters,
    }

# Threshold sweep
sweep_summary = []  # (r_name, den_name, thresh, P, R, F1, n_pred)

for (r_name, radius), (den_name, n_craters) in product(shallow_cases, density_sweep):
    data      = shallow_images[(r_name, den_name)]
    pred      = data["pred"]
    gt_coords = data["gt"]
    img_disp  = data["img_disp"]

    for thresh in THRESH_LIST:
        pred_coords = tmt.template_match_t(
            pred.copy(),
            minrad=MINRAD, maxrad=MAXRAD,
            longlat_thresh2=LONGLAT_THRESH2,
            rad_thresh=RAD_THRESH,
            template_thresh=thresh,
            target_thresh=TARGET_THRESH
        )

        matched_gt   = set()
        matched_pred = set()
        for pi, (px, py, pr) in enumerate(pred_coords):
            for gi, (gx, gy, gr) in enumerate(gt_coords):
                dist = np.sqrt((px - gx)**2 + (py - gy)**2)
                if dist < (pr + gr) * 0.5 and abs(pr - gr) < gr:
                    if gi not in matched_gt and pi not in matched_pred:
                        matched_gt.add(gi)
                        matched_pred.add(pi)

        TP = len(matched_gt)
        FP = len(pred_coords) - TP
        FN = len(gt_coords) - TP
        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        sweep_summary.append((r_name, den_name, thresh, precision, recall, f1, len(pred_coords)))

    # 畫這個條件的 threshold vs F1 折線圖
    row = [
        (t, f1) for (rn, dn, t, p, r, f1, _) in sweep_summary
        if rn == r_name and dn == den_name
    ]
    thresholds = [t for t, _ in row]
    f1s        = [f for _, f in row]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, f1s, marker="o", color="steelblue")
    ax.set_xlabel("Template Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(
        f"Shallow | {r_name} r={radius}px | density={den_name} n={n_craters}"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(
        os.path.join(shallow_dir, f"sweep_{r_name}_{den_name}.png"),
        dpi=100, bbox_inches="tight"
    )
    plt.close(fig)
    gc.collect()

# 印出 sweep 結果
print(f"\n  {'Size':<8} {'Density':<10} {'Thresh':>7} | {'Pred':>5} | {'P':>6} | {'R':>6} | {'F1':>6}")
print("-"*70)
for (r_name, den_name, thresh, p, r, f1, n_pred) in sweep_summary:
    marker = " <-- (原始設定)" if thresh == 0.35 else ""
    print(f"  {r_name:<8} {den_name:<10} {thresh:>7.2f} | {n_pred:>5} | "
          f"{p:>6.3f} | {r:>6.3f} | {f1:>6.3f}{marker}")

print(f"\n圖片輸出：{shallow_dir}")
