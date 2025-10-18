import os
import glob
import cv2
import numpy as np
from typing import Tuple

from facenet import FaceNet
from eigenfaces import Eigenfaces
from detectors import detect_faces
from robustness_eval import embed_images_facenet, knn_top1_chunked


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_gallery_from_indian(size=(100, 100)):
    """
    Build a gallery from the Indian dataset using saved artifacts.
    Returns:
        gallery_features (dict): { 'facenet': (emb, labels), 'eigenfaces': (proj, labels, model) }
        label_map (dict)
    """
    from preprocessing import createSplit
    root_dirs = [
        os.path.join('A1', 'indian-face-dataset', 'train'),
        os.path.join('A1', 'indian-face-dataset', 'val')
    ]
    X_train, X_test, y_train, y_test, label_map = createSplit(root_dirs, testSize=0.3, size=size)
    model_fn = FaceNet(128)
    model_fn.load(os.path.join('A1', 'results', 'facenet_model.pth'), device='cpu')
    model_fn.eval()
    fn_train_emb = embed_images_facenet(model_fn, X_train, size, batch_size=512)

    # eigenfaces gallery projections
    eig_model = Eigenfaces.load(os.path.join('A1', 'results', 'eigenfaces_model.npz'))
    eigen_train_proj = eig_model.project(X_train).astype(np.float32, copy=False)

    gallery = {
        'facenet': (fn_train_emb.astype(np.float32, copy=False), y_train, model_fn),
        'eigenfaces': (eigen_train_proj, y_train, eig_model)
    }
    return gallery, label_map


def crop_and_resize(img, box, size=(100, 100)):
    x, y, w, h = box
    x2, y2 = x+w, y+h
    x, y = max(0, x), max(0, y)
    crop = img[y:y2, x:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return gray.reshape(-1)


def recognize_face_facenet(model_fn: FaceNet, gallery_emb: np.ndarray, gallery_labels: np.ndarray, face_gray_flat: np.ndarray, size: Tuple[int,int], thresh: float = None):
    q = embed_images_facenet(model_fn, face_gray_flat[None, :], size, batch_size=1)[0:1]
    idx = np.argmin(np.sum((gallery_emb - q)**2, axis=1))
    pred = gallery_labels[idx]
    # optional threshold via distance (you can tune empirically)
    dist = float(np.sqrt(np.sum((gallery_emb[idx] - q[0])**2)))
    return pred, dist


def recognize_face_eigen(eig_model: Eigenfaces, gallery_proj: np.ndarray, gallery_labels: np.ndarray, face_gray_flat: np.ndarray):
    q = eig_model.project(face_gray_flat[None, :]).astype(np.float32, copy=False)
    idx = np.argmin(np.sum((gallery_proj - q)**2, axis=1))
    pred = gallery_labels[idx]
    dist = float(np.sqrt(np.sum((gallery_proj[idx] - q[0])**2)))
    return pred, dist


def main():
    crowd_dir = os.path.join('A1', 'crowd_test')  # put ≥ 50 crowd images here
    out_dir = os.path.join('A1', 'results', 'crowd')
    ensure_dir(out_dir)

    size = (100, 100)
    gallery, label_map = load_gallery_from_indian(size=size)
    model_fn = gallery['facenet'][2]

    # reverse label map for naming
    inv_label = {v: k for k, v in label_map.items()}

    # iterate crowd images
    results_rows = []
    labels_csv = os.path.join(crowd_dir, 'labels.csv')
    gt = {}
    if os.path.exists(labels_csv):
        import csv
        with open(labels_csv, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                gt[row['image']] = row['gt_name']

    for img_path in glob.glob(os.path.join(crowd_dir, '*.*')):
        img = cv2.imread(img_path)
        if img is None:
            continue
        boxes = detect_faces(img)
        annotated = img.copy()
        for b in boxes:
            face_flat = crop_and_resize(img, b, size=size)
            if face_flat is None:
                continue
            fn_pred, fn_dist = recognize_face_facenet(model_fn, gallery['facenet'][0], gallery['facenet'][1], face_flat, size)
            eg_pred, eg_dist = recognize_face_eigen(gallery['eigenfaces'][2], gallery['eigenfaces'][0], gallery['eigenfaces'][1], face_flat)
            fn_name = inv_label.get(int(fn_pred), str(fn_pred))
            eg_name = inv_label.get(int(eg_pred), str(eg_pred))

            x, y, w, h = b
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated, f"FN:{fn_name} ({fn_dist:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"EF:{eg_name} ({eg_dist:.2f})", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)

            gt_name = gt.get(os.path.basename(img_path), '')
            results_rows.append([os.path.basename(img_path), x, y, w, h, fn_name, fn_dist, eg_name, eg_dist, gt_name])

        # save annotated image
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), annotated)

    # write CSV
    import csv
    with open(os.path.join(out_dir, 'crowd_recognition_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image','x','y','w','h','facenet_name','facenet_dist','eigen_name','eigen_dist','gt_name'])
        w.writerows(results_rows)

    print('Crowd recognition results saved under', out_dir)


if __name__ == '__main__':
    main()
