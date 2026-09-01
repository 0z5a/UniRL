import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


class RotationTransform():
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


class IrregularMaskDataset():
    def __init__(self, path, set="train"):
        super().__init__()

        self.set = set
        if set == "train":
            self.mask_path_list = list((Path(path) / "train").glob("*png")) # 55116, 640 x 960
        elif set == "test":
            self.mask_path_list = list((Path(path) / "test").glob("*png")) # 12000, 512 x 512
        else:
            raise NotImplementedError()
    
    # image_size: (h, w)
    def get_mask(self, image_size):
        if self.set == "train":
            selected_mask_path = self.mask_path_list[np.random.randint(0, len(self.mask_path_list), 1)[0]]
            selected_mask = self.mask_augment(selected_mask_path.as_posix(), image_size)

            while selected_mask.sum() / (selected_mask.shape[0] * selected_mask.shape[1]) <= 0.05:
                selected_mask_path = self.mask_path_list[np.random.randint(0, len(self.mask_path_list), 1)[0]]
                selected_mask = self.mask_augment(selected_mask_path.as_posix(), image_size)
        else:
            selected_mask_path = self.mask_path_list[np.random.randint(0, len(self.mask_path_list), 1)[0]]
            # resize interpolation threshold 128
            # PIL.Image use size with (w, h)
            selected_mask = np.array(Image.open(selected_mask_path.as_posix()).resize((image_size[1], image_size[0])))
            # 0-1 matrix, 1 for mask
            # image_size
            selected_mask = torch.from_numpy((selected_mask >= 128).astype(np.uint8)).float()
        
        return selected_mask

    def mask_augment(self, mask_path, image_size):
        mask = 255 - cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, int(256 * 0.6), 255, cv2.THRESH_BINARY)

        kernel_size = np.random.randint(9, 50, 1)[0]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel)

        mask = Image.fromarray(mask)
        mask = transforms.Compose([
            transforms.RandomCrop(min(mask.size)),
            RotationTransform(angles=[-90, 0, 90, 180]),
            transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), fill=0),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])(mask)

        # 0-1 matrix, 1 for mask
        # image_size
        return (mask > 0.99).float().squeeze(0)


if __name__ == "__main__":
    mask_dataset = IrregularMaskDataset("/apdcephfs_gy2/share_302507476/0_public_datasets/image_inpainting/v1/irregular_mask", set="test")
    mask = mask_dataset.get_mask((32, 32))  # h x w
    print(mask.shape, mask.sum() / (mask.shape[0] * mask.shape[1]))
    Image.fromarray(mask.squeeze(0).numpy().astype(np.uint8) * 255).save("vis/images/mask.png")

# PYTHONPATH="." python3 hymm/data_kits/irregular_mask_dataset.py