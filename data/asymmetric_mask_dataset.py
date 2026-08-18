"""Path-matched TokenCut mask datasets with mask-aligned ImageNet transforms."""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def _image_path(dataset, index):
    if hasattr(dataset, "data"):
        data = dataset.data
        if hasattr(data, "iloc") and "filepath" in data.columns:
            relative_path = data.iloc[index].filepath
            if hasattr(dataset, "root") and hasattr(dataset, "base_folder"):
                return os.path.join(dataset.root, dataset.base_folder, relative_path)
            return relative_path
        if isinstance(data, (list, tuple, np.ndarray)):
            return data[index]
    if hasattr(dataset, "samples"):
        return dataset.samples[index][0]
    raise AttributeError(f"Cannot infer an image path from {type(dataset).__name__}.")


def _mask_candidates(image_path):
    normalized = os.fspath(image_path).replace("\\", "/")
    stem = os.path.splitext(normalized)[0]
    parts = [part for part in stem.split("/") if part]
    candidates = [stem, parts[-1]]
    for tail in (2, 3):
        if len(parts) >= tail:
            candidates.append("/".join(parts[-tail:]))
    for anchor in ("images", "cars_train", "cars_test"):
        if anchor in parts:
            position = parts.index(anchor)
            candidates.append("/".join(parts[position:]))
            if position + 1 < len(parts):
                candidates.append("/".join(parts[position + 1:]))
    return list(dict.fromkeys(candidate + ".npy" for candidate in candidates))


class PairedImageNetTransform:
    def __init__(self, train, image_size=224, crop_pct=0.875):
        self.train = train
        self.image_size = image_size
        self.resize_size = int(image_size / crop_pct)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, image, mask):
        mask_image = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
        mask_image = TF.resize(mask_image, image.size[::-1], interpolation=InterpolationMode.NEAREST)
        image = TF.resize(image, [self.resize_size, self.resize_size], interpolation=InterpolationMode.BICUBIC)
        mask_image = TF.resize(mask_image, [self.resize_size, self.resize_size], interpolation=InterpolationMode.NEAREST)
        if self.train:
            top = random.randint(0, self.resize_size - self.image_size)
            left = random.randint(0, self.resize_size - self.image_size)
            image = TF.crop(image, top, left, self.image_size, self.image_size)
            mask_image = TF.crop(mask_image, top, left, self.image_size, self.image_size)
            if random.random() < 0.5:
                image, mask_image = TF.hflip(image), TF.hflip(mask_image)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        else:
            image, mask_image = TF.center_crop(image, self.image_size), TF.center_crop(mask_image, self.image_size)
        image = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        return image, TF.to_tensor(mask_image).squeeze(0)


class AsymmetricMaskDataset(Dataset):
    def __init__(self, dataset, mask_root, train, image_size=224, crop_pct=0.875):
        self.dataset = dataset
        self.mask_root = os.path.expanduser(mask_root)
        self.train = train
        self.transform = PairedImageNetTransform(train, image_size, crop_pct)

    def __len__(self):
        return len(self.dataset)

    def _load_mask(self, index):
        image_path = _image_path(self.dataset, index)
        attempted = []
        for relative_path in _mask_candidates(image_path):
            path = os.path.join(self.mask_root, *relative_path.split("/"))
            attempted.append(path)
            if os.path.isfile(path):
                return np.load(path, allow_pickle=False).astype(np.float32)
        raise FileNotFoundError(f"No mask for {image_path}. Tried:\n  " + "\n  ".join(attempted))

    def __getitem__(self, index):
        image, target, uq_index = self.dataset[index]
        mask = self._load_mask(index)
        foreground, foreground_mask = self.transform(image, mask)
        if not self.train:
            return foreground, target, uq_index, foreground_mask
        global_view, _ = self.transform(image, mask)
        return [global_view, foreground], target, uq_index, foreground_mask


class MergedAsymmetricMaskDataset(Dataset):
    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset

    def __len__(self):
        return len(self.labelled_dataset) + len(self.unlabelled_dataset)

    def __getitem__(self, index):
        if index < len(self.labelled_dataset):
            image, label, uq_index, mask = self.labelled_dataset[index]
            is_labelled = 1
        else:
            image, label, uq_index, mask = self.unlabelled_dataset[index - len(self.labelled_dataset)]
            is_labelled = 0
        return image, label, uq_index, np.array([is_labelled]), mask
