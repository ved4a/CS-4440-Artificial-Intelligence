import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import torch

from eigenfaces import Eigenfaces
from facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocessing import createSplit
from robustness_transforms import (
    apply_to_batch, adjust_brightness_contrast,
    add_gaussian_noise, add_blur, add_jpeg_compression,
    add_rect_occlusion, add_bar_occlusion, overlay_mask,
)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def knn_top1_chunked(train_feats: np.ndarray, train_labels: np.ndarray, test_feats: np.ndarray, chunk: int = 256) -> np.ndarray:
    T = train_feats.astype(np.float32, copy=False)
    TL2 = np.sum(T*T, axis=1, keepdims=True).T  # [1, n_train]
    preds = []
    for i in range(0, test_feats.shape[0], chunk):
        Q = test_feats[i:i+chunk].astype(np.float32, copy=False)
        QL2 = np.sum(Q*Q, axis=1, keepdims=True)  # [m, 1]
        # D^2 = ||Q||^2 + ||T||^2 - 2 Q T^T
        G = Q @ T.T
        D2 = QL2 + TL2 - 2.0 * G
        idx = np.argmin(D2, axis=1)
        preds.append(train_labels[idx])
    return np.concatenate(preds, axis=0)


def embed_images_facenet(model: FaceNet, X_flat: np.ndarray, size: Tuple[int, int], batch_size: int = 256) -> np.ndarray:
    H, W = size
    X = (X_flat.astype(np.float32) / 255.0).reshape(-1, 1, H, W)
    embs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy(X[i:i+batch_size])
            embs.append(model(batch).cpu().numpy())
    return np.vstack(embs).astype(np.float32, copy=False)


def evaluate_eigenfaces_under(X_train, y_train, X_test, y_test, model_path, size, variant_name, transform_fn=None, transform_kwargs=None):
    model = Eigenfaces.load(model_path)

    X_test_var = X_test
    if transform_fn is not None:
        X_test_var = apply_to_batch(X_test, size, transform_fn, **(transform_kwargs or {}))

    preds = model.recognize(X_train, y_train, X_test_var)
    acc = accuracy_score(y_test, preds)
    return acc, preds


def evaluate_facenet_under(train_emb, test_imgs, y_train, y_test, model_path, size, variant_name, transform_fn=None, transform_kwargs=None):
    # load model to compute new test embeddings if needed
    device = 'cpu'
    model = FaceNet(embedding_size=128)
    model.load(model_path, device=device)
    model.eval()

    # apply transform on test images in image space
    X_test_var = test_imgs
    if transform_fn is not None:
        X_test_var = apply_to_batch(test_imgs, size, transform_fn, **(transform_kwargs or {}))

    # embed transformed test
    # reuse FaceDataset shape expectations: [N, 1, H, W] float32
    H, W = size
    Xf = X_test_var.astype(np.float32) / 255.0
    Xf = Xf.reshape(-1, 1, H, W)
    # batched forward
    embs = []
    bs = 256
    with torch.no_grad():
        for i in range(0, Xf.shape[0], bs):
            batch = torch.from_numpy(Xf[i:i+bs])
            embs.append(model(batch).cpu().numpy())
    test_emb_var = np.vstack(embs)

    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(train_emb, y_train)
    preds = clf.predict(test_emb_var)
    acc = accuracy_score(y_test, preds)
    return acc, preds


def plot_bar(results: Dict[str, float], title: str, save_path: str):
    labels = list(results.keys())
    vals = [results[k] for k in labels]
    plt.figure(figsize=(10, 4))
    bars = plt.bar(labels, [v*100 for v in vals], color='steelblue')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xticks(rotation=25, ha='right')
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v*100:.1f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(dataset_choice='indian'):
    """
    dataset_choice: 'indian' or 'celebA'
    For celebA, we will use attributes for bias analysis. For recognition pairing, this example keeps the existing Indian-faces pipeline for training; celebA can be used for subgroup analysis and robustness-only probes if labels per identity are not prepared.
    """
    results_dir = os.path.join('A1', 'results', 'robustness')
    ensure_dir(results_dir)

    # Load base data and models from previous task (Indian faces)
    root_dirs = [
        os.path.join('A1', 'indian-face-dataset', 'train'),
        os.path.join('A1', 'indian-face-dataset', 'val'),
    ]
    size = (100, 100)
    X_train, X_test, y_train, y_test, label_map = createSplit(root_dirs, testSize=0.3, size=size)

    # Baseline accuracies (optimized: precompute projections/embeddings and avoid Python loops)
    eigen_model_path = os.path.join('A1', 'results', 'eigenfaces_model.npz')
    from sklearn.metrics import accuracy_score
    eig_model = Eigenfaces.load(eigen_model_path)
    # Precompute train/test PCA projections once
    eigen_train_feats = eig_model.project(X_train).astype(np.float32, copy=False)
    eigen_test_feats = eig_model.project(X_test).astype(np.float32, copy=False)
    base_preds_eig = knn_top1_chunked(eigen_train_feats, y_train, eigen_test_feats, chunk=256)
    base_acc_eig = accuracy_score(y_test, base_preds_eig)

    # FaceNet baseline uses saved embeddings
    facenet_model_path = os.path.join('A1', 'results', 'facenet_model.pth')
    train_emb_path = os.path.join('A1', 'results', 'facenet_train_embeddings.npy')
    test_emb_path = os.path.join('A1', 'results', 'facenet_test_embeddings.npy')
    # Load FaceNet once and reuse
    model_fn = FaceNet(embedding_size=128)
    model_fn.load(facenet_model_path, device='cpu')
    model_fn.eval()
    if os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        train_emb = np.load(train_emb_path).astype(np.float32, copy=False)
        test_emb = np.load(test_emb_path).astype(np.float32, copy=False)
    else:
        train_emb = embed_images_facenet(model_fn, X_train, size, batch_size=256)
        test_emb = embed_images_facenet(model_fn, X_test, size, batch_size=256)
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(train_emb, y_train)
    base_preds_fn = knn.predict(test_emb)
    base_acc_fn = accuracy_score(y_test, base_preds_fn)

    # bucket 1: lighting changes
    lighting_configs = {
        'bright_+30': dict(transform_fn=adjust_brightness_contrast, transform_kwargs={'alpha': 1.0, 'beta': 30}),
        'dark_-30': dict(transform_fn=adjust_brightness_contrast, transform_kwargs={'alpha': 1.0, 'beta': -30}),
        'high_contrast_1.5': dict(transform_fn=adjust_brightness_contrast, transform_kwargs={'alpha': 1.5, 'beta': 0}),
        'low_contrast_0.7': dict(transform_fn=adjust_brightness_contrast, transform_kwargs={'alpha': 0.7, 'beta': 0}),
    }

    lighting_results_eig = {}
    lighting_results_fn = {}
    for name, cfg in lighting_configs.items():
        X_test_var = apply_to_batch(X_test, size, cfg['transform_fn'], **cfg.get('transform_kwargs', {}))
        # reuse precomputed eigen_train_feats; only project transformed test
        eigen_test_var = eig_model.project(X_test_var).astype(np.float32, copy=False)
        preds_eig = knn_top1_chunked(eigen_train_feats, y_train, eigen_test_var, chunk=256)
        acc_eig = accuracy_score(y_test, preds_eig)
        lighting_results_eig[name] = acc_eig

        # FaceNet: recompute embeddings for transformed test
        test_emb_var = embed_images_facenet(model_fn, X_test_var, size, batch_size=256)
        preds_fn = knn.predict(test_emb_var)
        acc_fn = accuracy_score(y_test, preds_fn)
        lighting_results_fn[name] = acc_fn

    plot_bar({'baseline': base_acc_eig, **lighting_results_eig}, 'Eigenfaces: Lighting robustness', os.path.join(results_dir, 'eig_lighting.png'))
    plot_bar({'baseline': base_acc_fn, **lighting_results_fn}, 'FaceNet: Lighting robustness', os.path.join(results_dir, 'fn_lighting.png'))

    # bucket 2: image quality variations
    quality_configs = {
        'gauss_noise_sigma10': dict(fn=add_gaussian_noise, kw={'sigma': 10}),
        'gauss_noise_sigma25': dict(fn=add_gaussian_noise, kw={'sigma': 25}),
        'blur_k5': dict(fn=add_blur, kw={'ksize': 5}),
        'blur_k11': dict(fn=add_blur, kw={'ksize': 11}),
        'jpeg_q30': dict(fn=add_jpeg_compression, kw={'quality': 30}),
        'jpeg_q10': dict(fn=add_jpeg_compression, kw={'quality': 10}),
    }
    quality_results_eig = {}
    quality_results_fn = {}
    for name, cfg in quality_configs.items():
        X_test_var = apply_to_batch(X_test, size, cfg['fn'], **cfg['kw'])
        eigen_test_var = eig_model.project(X_test_var).astype(np.float32, copy=False)
        preds_eig = knn_top1_chunked(eigen_train_feats, y_train, eigen_test_var, chunk=256)
        acc_eig = accuracy_score(y_test, preds_eig)
        quality_results_eig[name] = acc_eig

        # FaceNet transformed embeddings
        test_emb_var = embed_images_facenet(model_fn, X_test_var, size, batch_size=256)
        acc_fn = accuracy_score(y_test, knn.predict(test_emb_var))
        quality_results_fn[name] = acc_fn

    plot_bar({'baseline': base_acc_eig, **quality_results_eig}, 'Eigenfaces: Quality robustness', os.path.join(results_dir, 'eig_quality.png'))
    plot_bar({'baseline': base_acc_fn, **quality_results_fn}, 'FaceNet: Quality robustness', os.path.join(results_dir, 'fn_quality.png'))

    # bucket 3: occlusions
    occ_configs = {
        'rect_center_20%': dict(fn=add_rect_occlusion, kw={'occ_frac': 0.2, 'mode': 'center'}),
        'rect_random_30%': dict(fn=add_rect_occlusion, kw={'occ_frac': 0.3, 'mode': 'random'}),
        'bar_eyes': dict(fn=add_bar_occlusion, kw={'thickness': 12, 'position': 'eyes'}),
        'bar_mouth': dict(fn=add_bar_occlusion, kw={'thickness': 12, 'position': 'mouth'}),
        'mask_bottom40%': dict(fn=overlay_mask, kw={'ratio': 0.4}),
    }
    occ_results_eig = {}
    occ_results_fn = {}
    for name, cfg in occ_configs.items():
        X_test_var = apply_to_batch(X_test, size, cfg['fn'], **cfg['kw'])
        eigen_test_var = eig_model.project(X_test_var).astype(np.float32, copy=False)
        preds_eig = knn_top1_chunked(eigen_train_feats, y_train, eigen_test_var, chunk=256)
        acc_eig = accuracy_score(y_test, preds_eig)
        occ_results_eig[name] = acc_eig

        test_emb_var = embed_images_facenet(model_fn, X_test_var, size, batch_size=256)
        acc_fn = accuracy_score(y_test, knn.predict(test_emb_var))
        occ_results_fn[name] = acc_fn

    plot_bar({'baseline': base_acc_eig, **occ_results_eig}, 'Eigenfaces: Occlusion robustness', os.path.join(results_dir, 'eig_occlusion.png'))
    plot_bar({'baseline': base_acc_fn, **occ_results_fn}, 'FaceNet: Occlusion robustness', os.path.join(results_dir, 'fn_occlusion.png'))

    # bucket 4: explainability
    # - save mean face and first eigenfaces already done in eigenfaces_main; re-copy or just ensure present
    # - also show reconstructions at different k using the saved model
    H, W = size
    k_list = [5, 20, min(50, eig_model.eigenfaces.shape[1]-1)]
    sample_idx = np.random.choice(X_test.shape[0], size=min(8, X_test.shape[0]), replace=False)
    ensure_dir(os.path.join(results_dir, 'explainability'))
    for idx in sample_idx:
        x = X_test[idx:idx+1]
        coeffs = eig_model.project(x)
        # progressive reconstructions by truncating coeffs
        for k in k_list:
            c_trunc = coeffs[:, :k]
            # Use matching top-k eigenfaces for reconstruction to avoid dimension mismatch
            E_k = eig_model.eigenfaces[:, :k]
            recon = (c_trunc @ E_k.T) + eig_model.meanFace
            recon = recon.reshape(H, W)
            orig = x.reshape(H, W)
            vis = np.hstack([
                (orig - orig.min())/(orig.max()-orig.min()+1e-8),
                (recon - recon.min())/(recon.max()-recon.min()+1e-8)
            ])
            out = (vis*255).astype(np.uint8)
            cv2.imwrite(os.path.join(results_dir, 'explainability', f'eig_recon_idx{idx}_k{k}.png'), out)

    # bucket 5: bias analysis
    # 5a) indian faces: brightness tertiles as simple subgrouping (proxy for illumination/skin tone)
    intensities = X_test.mean(axis=1)
    q1, q2 = np.quantile(intensities, [1/3, 2/3])
    idx_low = np.where(intensities <= q1)[0]
    idx_mid = np.where((intensities > q1) & (intensities <= q2))[0]
    idx_high = np.where(intensities > q2)[0]
    per_group_eig = {
        'low': float('nan') if len(idx_low)==0 else accuracy_score(y_test[idx_low], np.array(base_preds_eig)[idx_low]),
        'mid': float('nan') if len(idx_mid)==0 else accuracy_score(y_test[idx_mid], np.array(base_preds_eig)[idx_mid]),
        'high': float('nan') if len(idx_high)==0 else accuracy_score(y_test[idx_high], np.array(base_preds_eig)[idx_high]),
    }
    per_group_fn = {
        'low': float('nan') if len(idx_low)==0 else accuracy_score(y_test[idx_low], np.array(base_preds_fn)[idx_low]),
        'mid': float('nan') if len(idx_mid)==0 else accuracy_score(y_test[idx_mid], np.array(base_preds_fn)[idx_mid]),
        'high': float('nan') if len(idx_high)==0 else accuracy_score(y_test[idx_high], np.array(base_preds_fn)[idx_high]),
    }
    # save CSV
    with open(os.path.join(results_dir, 'indian_bias_brightness.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'eigenfaces_acc', 'facenet_acc'])
        for g in ['low','mid','high']:
            w.writerow([g, per_group_eig[g], per_group_fn[g]])
    # plot grouped bars
    plt.figure(figsize=(6,4))
    x = np.arange(3)
    width = 0.35
    eig_vals = np.array([per_group_eig['low'], per_group_eig['mid'], per_group_eig['high']]) * 100.0
    fn_vals = np.array([per_group_fn['low'], per_group_fn['mid'], per_group_fn['high']]) * 100.0
    plt.bar(x - width/2, eig_vals, width, label='Eigenfaces')
    plt.bar(x + width/2, fn_vals, width, label='FaceNet')
    plt.xticks(x, ['low','mid','high'])
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by brightness tertiles (Indian faces)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'indian_bias_brightness.png'), dpi=150)
    plt.close()

    # 5b) CelebA attributes example (gender self-consistency proxy)
    # minimal: group by Male attribute and compare accuracy using CelebA subset if available
    try:
        celeb_dir = os.path.join('A1', 'celeb-a')
        attr_path = os.path.join(celeb_dir, 'list_attr_celeba.csv')
        if os.path.exists(attr_path):
            # Read image_id and Male columns using csv
            image_ids = []
            male_flags = []
            with open(attr_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_ids.append(row['image_id'])
                    # Values are -1 or 1 in CelebA
                    try:
                        male_flags.append(1 if int(row['Male']) == 1 else 0)
                    except Exception:
                        # if already 0/1 strings
                        male_flags.append(1 if str(row['Male']).strip() in ('1', 'True', 'true') else 0)
            # Build simple lists for sampling
            male_ids = [iid for iid, m in zip(image_ids, male_flags) if m == 1]
            female_ids = [iid for iid, m in zip(image_ids, male_flags) if m == 0]

            model = FaceNet(128)
            model.load(facenet_model_path, device='cpu')
            model.eval()

            def embed_img(gray):
                g = gray.astype(np.float32)/255.0
                g = torch.from_numpy(g[None, None, :, :])
                with torch.no_grad():
                    e = model(g).cpu().numpy()[0]
                return e

            N = 200
            imgs_dir = os.path.join(celeb_dir, 'img_align_celeba')
            rng = np.random.default_rng(42)
            male_sample = rng.choice(male_ids, size=min(N, len(male_ids)), replace=False) if male_ids else []
            female_sample = rng.choice(female_ids, size=min(N, len(female_ids)), replace=False) if female_ids else []

            def group_consistency(file_list, transform):
                sims = []
                for fname in file_list:
                    f = os.path.join(imgs_dir, fname)
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        continue
                    im = cv2.resize(im, size)
                    e0 = embed_img(im)
                    im_t = transform(im)
                    e1 = embed_img(im_t)
                    s = np.dot(e0, e1)/(np.linalg.norm(e0)*np.linalg.norm(e1)+1e-8)
                    sims.append(s)
                return np.array(sims)

            tf = lambda im: add_jpeg_compression(add_gaussian_noise(im, sigma=15), quality=25)
            sims_m = group_consistency(male_sample, tf)
            sims_f = group_consistency(female_sample, tf)

            # Write CSV manually
            out_csv = os.path.join(results_dir, 'celeba_self_consistency_cosine.csv')
            with open(out_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['group', 'cosine_similarity'])
                for s in sims_m:
                    w.writerow(['male', float(s)])
                for s in sims_f:
                    w.writerow(['female', float(s)])

            # boxplot
            plt.figure(figsize=(4,4))
            data = [sims_m if len(sims_m)>0 else [np.nan], sims_f if len(sims_f)>0 else [np.nan]]
            plt.boxplot(data, tick_labels=['male','female'])
            plt.ylabel('Cosine sim(orig, transformed)')
            plt.title('FaceNet self-consistency by gender (CelebA)')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'celeba_gender_consistency.png'), dpi=150)
            plt.close()
    except Exception as e:
        print('Bias analysis skipped due to error:', e)

    # save tabular deltas for each bucket
    def write_delta_csv(baseline, dct, out_csv):
        rows = []
        for k, v in dct.items():
            rows.append((k, float(v), float((v - baseline) * 100.0)))
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['variant', 'accuracy', 'delta_pct'])
            w.writerows(rows)

    write_delta_csv(base_acc_eig, lighting_results_eig, os.path.join(results_dir, 'eigenfaces_lighting.csv'))
    write_delta_csv(base_acc_fn, lighting_results_fn, os.path.join(results_dir, 'facenet_lighting.csv'))
    write_delta_csv(base_acc_eig, quality_results_eig, os.path.join(results_dir, 'eigenfaces_quality.csv'))
    write_delta_csv(base_acc_fn, quality_results_fn, os.path.join(results_dir, 'facenet_quality.csv'))
    write_delta_csv(base_acc_eig, occ_results_eig, os.path.join(results_dir, 'eigenfaces_occlusion.csv'))
    write_delta_csv(base_acc_fn, occ_results_fn, os.path.join(results_dir, 'facenet_occlusion.csv'))

    print('Robustness evaluation artifacts saved under', results_dir)


if __name__ == '__main__':
    main('indian')
