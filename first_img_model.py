import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ===== 路徑設定 =====
dev_images_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\dev_images.hdf5"
model_path = r"C:\Users\jhan3\OneDrive\桌面\大二專題\DeepMoon-master\model_keras2.h5"

# ===== 讀取模型 =====
model = tf.keras.models.load_model(model_path, compile=False)

# ===== 讀取 dev_images[0] 和 target_masks[0] =====
idx = 0
with h5py.File(dev_images_path, "r") as f:
    img = f["input_images"][idx]
    mask = f["target_masks"][idx]

print("img shape =", img.shape)
print("img dtype =", img.dtype)
print("img min/max/mean =", img.min(), img.max(), img.mean())

print("mask shape =", mask.shape)
print("mask dtype =", mask.dtype)
print("mask min/max/mean =", mask.min(), mask.max(), mask.mean())

# ===== 前處理：per-image rescale =====
img_float = img.astype("float32")
img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

# ===== 加強後影像：CLAHE =====
try:
    from skimage import exposure
    img_enhanced = exposure.equalize_adapthist(img_norm, clip_limit=0.03)
except ModuleNotFoundError:
    print("沒有安裝 scikit-image，改用簡單 contrast stretch")
    p2, p98 = np.percentile(img_norm, (2, 98))
    img_enhanced = np.clip((img_norm - p2) / (p98 - p2 + 1e-8), 0, 1)

# ===== 丟進模型 =====
x = img_norm.reshape(1, 256, 256, 1)
pred_raw = model.predict(x)

print("pred_raw shape =", pred_raw.shape)

if pred_raw.ndim == 4:
    pred = pred_raw[0, :, :, 0]
elif pred_raw.ndim == 3:
    pred = pred_raw[0, :, :]
else:
    raise ValueError(f"模型輸出維度異常：{pred_raw.shape}")

print("pred min/max/mean =", pred.min(), pred.max(), pred.mean())

# =========================================================
# 找隕石坑（用 heatmap -> binary mask -> Hough circle）
# =========================================================
try:
    from skimage.morphology import remove_small_objects
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks

    # 1. 二值化
    threshold = 0.25 * pred.max()   # 你之後可以自己調，例如 0.2~0.4
    binary_map = pred > threshold

    # 2. 去除太小雜訊
    binary_map = remove_small_objects(binary_map, min_size=20)

    # 3. 邊緣
    edges = canny(binary_map.astype(float), sigma=1)

    # 4. Hough 找圓
    radii_range = np.arange(5, 60, 2)
    hough_res = hough_circle(edges, radii_range)

    accums, cx, cy, radii = hough_circle_peaks(
        hough_res,
        radii_range,
        total_num_peaks=15
    )

except ModuleNotFoundError:
    print("缺少 scikit-image，無法做 circle detection。請先安裝：pip install scikit-image")
    binary_map = pred > (0.25 * pred.max())
    cx, cy, radii = [], [], []

print("找到的圓數量 =", len(radii))

# =========================================================
# 畫圖：上四下三
# =========================================================
fig, ax = plt.subplots(2, 4, figsize=(20, 10))

# ===== 上排四張 =====
ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].set_title(f"dev image[{idx}]")
ax[0, 0].axis("off")

ax[0, 1].imshow(img_enhanced, cmap="gray")
ax[0, 1].set_title("enhanced image")
ax[0, 1].axis("off")

ax[0, 2].imshow(mask, cmap="gray")
ax[0, 2].set_title(f"target mask[{idx}]")
ax[0, 2].axis("off")

ax[0, 3].imshow(pred, cmap="hot")
ax[0, 3].set_title("model heatmap")
ax[0, 3].axis("off")

# ===== 下排三張 =====
ax[1, 0].imshow(img, cmap="gray")
ax[1, 0].imshow(pred, cmap="hot", alpha=0.5)
ax[1, 0].set_title("heatmap overlay")
ax[1, 0].axis("off")

ax[1, 1].imshow(binary_map, cmap="gray")
ax[1, 1].set_title("binary mask")
ax[1, 1].axis("off")

ax[1, 2].imshow(img, cmap="gray")
ax[1, 2].set_title(f"detected craters ({len(radii)})")
ax[1, 2].axis("off")

for x0, y0, r in zip(cx, cy, radii):
    circle = plt.Circle((x0, y0), r, color="red", fill=False, linewidth=2)
    ax[1, 2].add_patch(circle)

# 最後一格關掉，變成上四下三
ax[1, 3].axis("off")

plt.tight_layout()
plt.show()
