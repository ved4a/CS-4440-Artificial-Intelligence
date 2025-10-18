import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)

def classification_summary(y_true, y_pred, labels=None, print_report=True):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    if print_report:
        print(report)
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "report_text": report,
    }

def _pairwise_euclidean(A, B):
    # A: [n_a, d], B: [n_b, d]
    A2 = np.sum(A*A, axis=1, keepdims=True)
    B2 = np.sum(B*B, axis=1, keepdims=True).T
    return np.sqrt(np.clip(A2 + B2 - 2*np.dot(A, B.T), a_min=0, a_max=None))

def rank_k_identification(train_feats, train_labels, test_feats, y_true, ks=(1, 5)):
    D = _pairwise_euclidean(test_feats, train_feats)  # [n_test, n_train]
    nn_order = np.argsort(D, axis=1)  # ascending distances
    results = {}
    for k in ks:
        topk_labels = np.take(train_labels, nn_order[:, :k])
        hit = np.any(topk_labels == y_true[:, None], axis=1)
        results[f"rank_{k}"] = np.mean(hit)
    # also return the top-1 predictions used for standard accuracy
    top1_pred = np.take(train_labels, nn_order[:, 0])
    results["top1_pred"] = top1_pred
    return results