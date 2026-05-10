"""
火星 Zero-Shot 評估腳本
使用月球訓練的 DeepMoon 模型，直接在 DeepMars 測試資料上評估
"""

import numpy as np
import h5py
import pandas as pd
from keras.models import load_model
from skimage.feature import match_template
import cv2

# ===================== 路徑設定（請自行修改） =====================
MODEL_PATH = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"
IMAGES_HDF5 = r"C:\Users\jhan3\OneDrive\桌面\大二專題\deepmars-master\tests\ran_images_175000.hdf5"
CRATERS_HDF5 = r"C:\Users\jhan3\OneDrive\桌面\大二專題\deepmars-master\tests\ran_craters_175000.hdf5"

# ===================== 參數設定 =====================
THRESHOLD = 0.35        # 從月球最佳值開始，可調整
TEMPLATE_MATCH_T = 0.5  # template matching threshold
MIN_MATCH_DIST = 0.5    # 隕石坑匹配距離（相對於半徑）

# ===================== 讀取資料 =====================
def load_data(images_path, craters_path):
    with h5py.File(images_path, 'r') as f:
        images = f['input_images'][:]          # shape: (N, 256, 256)
        masks  = f['target_masks'][:]          # shape: (N, 256, 256)
        img_ids = sorted([k for k in f['cll_xy'].keys()])

    craters = {}
    with h5py.File(craters_path, 'r') as f:
        for img_id in img_ids:
            if img_id in f:
                cols = [x.decode() for x in f[img_id]['axis0'][:]]
                vals = f[img_id]['block0_values'][:]
                df = pd.DataFrame(vals, columns=cols)
                craters[img_id] = df

    return images, masks, img_ids, craters


# ===================== Template Matching 後處理 =====================
def template_match_t(pred_mask, threshold=TEMPLATE_MATCH_T):
    """
    從預測 mask 中用 template matching 找出隕石坑圓心和半徑
    回傳: list of (x, y, r)
    """
    detected = []
    binary = (pred_mask > threshold).astype(np.uint8)

    # 嘗試不同半徑的圓形 template
    for r in range(3, 40):
        # 建立圓環 template
        size = 2 * r + 3
        template = np.zeros((size, size), dtype=np.float32)
        cv2.circle(template, (r+1, r+1), r, 1, 1)
        template = template / template.sum()

        if template.shape[0] > pred_mask.shape[0] or template.shape[1] > pred_mask.shape[1]:
            continue

        result = match_template(pred_mask.astype(np.float32), template, pad_input=True)
        peaks = np.argwhere(result > TEMPLATE_MATCH_T)

        for peak in peaks:
            y, x = peak
            detected.append((x, y, r))

    # 去除重複（NMS）
    detected = non_max_suppression(detected)
    return detected


def non_max_suppression(detections, overlap_thresh=0.5):
    if not detections:
        return []

    detections = np.array(detections, dtype=np.float32)
    x, y, r = detections[:, 0], detections[:, 1], detections[:, 2]

    keep = []
    idxs = np.argsort(r)[::-1]  # 大圓優先

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        dist = np.sqrt((x[idxs[1:]] - x[i])**2 + (y[idxs[1:]] - y[i])**2)
        overlap = dist < (r[i] * overlap_thresh + r[idxs[1:]] * overlap_thresh)
        idxs = idxs[1:][~overlap]

    return detections[keep].tolist()


# ===================== 評估指標 =====================
def match_craters(detected, ground_truth, match_thresh=MIN_MATCH_DIST):
    """
    比對預測和 ground truth 隕石坑
    match_thresh: 距離 < match_thresh * r 視為正確匹配
    """
    if len(detected) == 0:
        return 0,0, len(ground_truth)

    tp = 0
    matched_gt = set()

    for det in detected:
        dx, dy, dr = det[0], det[1], det[2]
        best_dist = float('inf')
        best_j = -1

        for j, gt in ground_truth.iterrows():
            if j in matched_gt:
                continue
            dist = np.sqrt((dx - gt['x'])**2 + (dy - gt['y'])**2)
            norm_dist = dist / (gt['Diameter (pix)'] / 2)
            if norm_dist < best_dist:
                best_dist = norm_dist
                best_j = j

        if best_dist < match_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)

    fp = len(detected) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


# ===================== 主程式 =====================
def main():
    print("載入模型...")
    model = load_model(MODEL_PATH,compile=False)

    print("載入火星資料...")
    images, masks, img_ids, craters = load_data(IMAGES_HDF5, CRATERS_HDF5)

    print(f"共 {len(images)} 張圖，{len(img_ids)} 個有標注的圖")
    print(f"Threshold: {THRESHOLD}\n")

    total_tp, total_fp, total_fn = 0, 0, 0
    results = []

    for i, img_id in enumerate(img_ids):
        if i >= len(images):
            break

        # 推論
        img = images[i].astype(np.float32) / 255.0
        img_input = img.reshape(1, 256, 256, 1)
        pred = model.predict(img_input, verbose=0)[0, :, :]

        # 後處理
        detected = template_match_t(pred, threshold=THRESHOLD)

        # 取得 ground truth
        gt = craters.get(img_id, pd.DataFrame())

        if len(gt) == 0:
            print(f"[{img_id}] 無 ground truth，跳過")
            continue

        # 計算指標
        tp, fp, fn = match_craters(detected, gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        results.append({
            'img_id': img_id,
            'detected': len(detected),
            'gt_count': len(gt),
            'TP': tp, 'FP': fp, 'FN': fn,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })

        print(f"[{img_id}] detected={len(detected)}, gt={len(gt)}, "
              f"TP={tp}, FP={fp}, FN={fn} | "
              f"P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

    # 整體結果
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1   = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    print("\n" + "="*50)
    print("火星 Zero-Shot 評估結果")
    print("="*50)
    print(f"Threshold:  {THRESHOLD}")
    print(f"Precision:  {overall_prec:.4f}")
    print(f"Recall:     {overall_rec:.4f}")
    print(f"F1:         {overall_f1:.4f}")
    print("="*50)

    # 跟月球結果比較
    print("\n對比月球最佳結果（threshold=0.35）：")
    print(f"  月球 Precision: 0.7631 → 火星: {overall_prec:.4f}")
    print(f"  月球 Recall:    0.2361 → 火星: {overall_rec:.4f}")
    print(f"  月球 F1:        0.3606 → 火星: {overall_f1:.4f}")

    return results


if __name__ == '__main__':
    main()
