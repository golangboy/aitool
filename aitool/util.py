import numpy as np
import torch
import os
from PIL import Image
import shutil


def visual_mask(img_np: np.ndarray, labels_name: list):
    assert len(img_np.shape) == 2
    assert len(np.unique(img_np)) <= len(labels_name)
    unique_labels = np.unique(img_np)
    np.random.seed(0)
    output_rgb = np.zeros(
        (img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
    for label, label_name in zip(unique_labels, labels_name):
        points = np.where(img_np == label)
        label_color = [np.random.randint(100, 255) for _ in range(3)]
        if label == 0:
            label_color = [0, 0, 0]
        for point in zip(points[0], points[1]):
            output_rgb[point[0], point[1], :] = label_color
        pass
    return output_rgb


def split_dataset(data_dir: str, t=0.6, new_train_dir="new_train", new_val_dir="new_val"):
    assert data_dir is not None
    images_files = os.listdir(os.path.join(data_dir, 'images'))
    masks_files = os.listdir(os.path.join(data_dir, 'masks'))
    assert len(images_files) == len(masks_files)

    # 同时打乱images_files和masks_files
    seed = int(np.random.rand()*1000)

    np.random.seed(seed)
    np.random.shuffle(images_files)
    np.random.seed(seed)
    np.random.shuffle(masks_files)

    data_length = len(images_files)
    train_size = int(data_length*t)
    train_images = images_files[:train_size]
    train_masks = masks_files[:train_size]
    val_images = images_files[train_size:]
    val_masks = masks_files[train_size:]

    os.makedirs(os.path.join(new_train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_train_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, 'masks'), exist_ok=True)

    for file in train_images:
        shutil.copy(os.path.join(data_dir, 'images', file),
                    os.path.join(new_train_dir, 'images', file))
    for file in train_masks:
        shutil.copy(os.path.join(data_dir, 'masks', file),
                    os.path.join(new_train_dir, 'masks', file))

    for file in val_images:
        shutil.copy(os.path.join(data_dir, 'images', file),
                    os.path.join(new_val_dir, 'images', file))
    for file in val_masks:
        shutil.copy(os.path.join(data_dir, 'masks', file),
                    os.path.join(new_val_dir, 'masks', file))
    pass
