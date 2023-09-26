import numpy as np
import torch


"""
b是预测
hist[实际上的值][模型输出的值]
"""


def fast_hist(a: np.ndarray, b: np.ndarray, n: int):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist: np.ndarray):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def visual_mask(img_np: np.ndarray, labels_name: list):
    assert len(img_np.shape) == 2
    assert len(np.unique(img_np)) <= len(labels_name)
    unique_labels = np.unique(img_np)
    np.random.seed(0)
    output_rgb = np.zeros(
        (img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
    for label, label_name in zip(unique_labels, labels_name):
        np.random.seed(label)
        points = np.where(img_np == label)
        label_color = [np.random.randint(100, 255) for _ in range(3)]
        if label == 0:
            label_color = [0, 0, 0]
        for point in zip(points[0], points[1]):
            output_rgb[point[0], point[1], :] = label_color
        pass
    return output_rgb
