import cv2
import numpy as np


# lighting (brightness/contrast)
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


# quality variations
def add_gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(img, ksize=5):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_jpeg_compression(img, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    if not result:
        return img
    dec = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    return dec


# occlusions (rectangle)
def add_rect_occlusion(img, occ_frac=0.2, mode='center'):
    h, w = img.shape[:2]
    occ_h, occ_w = int(h*occ_frac), int(w*occ_frac)
    if mode == 'random':
        y = np.random.randint(0, max(1, h - occ_h))
        x = np.random.randint(0, max(1, w - occ_w))
    else:  # center
        y = (h - occ_h) // 2
        x = (w - occ_w) // 2
    out = img.copy()
    out[y:y+occ_h, x:x+occ_w] = 0
    return out

# occulusions (bar)
def add_bar_occlusion(img, thickness=10, position='eyes'):
    h, w = img.shape[:2]
    out = img.copy()
    if position == 'eyes':
        y = h // 3
        out[max(0, y-thickness//2):min(h, y+thickness//2), :] = 0
    elif position == 'mouth':
        y = 2*h // 3
        out[max(0, y-thickness//2):min(h, y+thickness//2), :] = 0
    else:
        x = w // 2
        out[:, max(0, x-thickness//2):min(w, x+thickness//2)] = 0
    return out

# simulate application of a mask
def overlay_mask(img, ratio=0.4):
    h, w = img.shape[:2]
    y0 = int(h * (1.0 - ratio))
    out = img.copy()
    out[y0:, :] = 0
    return out


# batch helpers
def apply_to_batch(batch_flat, size, fn, **kwargs):
    H, W = size
    out = []
    for i in range(batch_flat.shape[0]):
        img = batch_flat[i].reshape(H, W)
        out_img = fn(img, **kwargs)
        out.append(out_img.reshape(-1))
    return np.stack(out, axis=0)
