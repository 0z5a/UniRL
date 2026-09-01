import random
from pathlib import Path

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

from hymm.data_kits.index_dataset import IndexDataset
from hymm.utils.helpers import to_2tuple


def image_reader(source):
    with open(source, "rb") as fp:
        binary = fp.read()
    width, height = Image.open(source).size
    md5 = md5sum_binary(binary)
    return binary, height, width, md5


def imagenet_1k_to_arrow(src_dir, label_csv, val_label_csv):
    train_dir = Path(src_dir) / "train"
    val_dir = Path(src_dir) / "val"
    label_data = pd.read_csv(label_csv, header=0)
    train_label_dict = label_data.set_index('id')['label'].to_dict()
    label_name_dict = label_data.set_index('label')['name'].to_dict()
    label_name_cn_dict = label_data.set_index('label')['name_cn'].to_dict()
    val_labels = Path(val_label_csv).read_text().splitlines()
    val_mapping = [x.split(' ')
                   for x in (Path(val_label_csv).parent / "ILSVRC2012_mapping.txt").read_text().splitlines()]
    val_mapping = {val_id: train_label_dict[train_id] for val_id, train_id in val_mapping[:1000]}
    val_label_dict = {i: val_mapping[val_id] for i, val_id in enumerate(val_labels, start=1)}

    def train_data_provider():
        for sub_dir in sorted(train_dir.glob('*'), key=lambda x: train_label_dict[x.name]):
            label = train_label_dict[sub_dir.name]
            data = []
            for img in sorted(sub_dir.glob('*')):
                data.append({
                    "image_path_absolute": str(img),
                    "label": label,
                    "text_source": label_name_dict[label],
                    "text_trans": label_name_cn_dict[label],
                })
            yield pd.DataFrame(data)

    def val_data_provider():
        data = []
        for img in sorted(val_dir.glob('*.JPEG')):
            label = val_label_dict[int(img.stem.split('_')[-1])]
            data.append({
                "image_path_absolute": str(img),
                "label": label,
                "text_source": label_name_dict[label],
                "text_trans": label_name_cn_dict[label],
            })
        yield pd.DataFrame(data)

    make_arrows(train_data_provider(),
                save_dir="/apdcephfs_nj10/share_301739632/0_public_datasets/imagenet_1k/train",
                image_source_col="image_path_absolute",
                image_target_col=["image", "height", "width", "md5"],
                num_slice=None,
                reader=image_reader,
                remove_source_col=False,
                )

    make_arrows(val_data_provider(),
                save_dir="/apdcephfs_nj10/share_301739632/0_public_datasets/imagenet_1k/val",
                image_source_col="image_path_absolute",
                image_target_col=["image", "height", "width", "md5"],
                num_slice=5000,
                reader=image_reader,
                remove_source_col=False,
                )


class ImageNetDataset(IndexDataset):
    def __init__(self,
                 image_size,
                 index_file,
                 crop_type="random",    # random/center
                 uncond_p=0,
                 logger=None,
                 ):
        super().__init__(index_file, logger=logger)
        self.image_size = to_2tuple(image_size)
        self.crop_type = crop_type
        self.uncond_p = uncond_p

        self.index_manager = self.load_index()

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.tensor_to_pil_image = transforms.Compose(
            [
                transforms.Normalize([-1], [2]),
                transforms.ToPILImage(),
            ]
        )

    def get_raw_image(self, index, image_key="image"):
        try:
            ret = self.index_manager.get_image(index, image_key)
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"{type(e)}: {e}")
            ret = Image.new("RGB", (self.image_size[0], self.image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag

    def get_image_with_size(self, index):
        image, image_flag = self.get_raw_image(index, image_key="image")

        origin_size = list(image.size)[::-1]  # (h_ori, w_ori)
        target_size = self.image_size[0], self.image_size[1]

        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type=self.crop_type, resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image_tensor, kwargs, image_flag

    def get_label(self, index):
        return self.index_manager.get_attribute(index, 'label')

    def __getitem__(self, index):
        image_tensor, kwargs, image_flag = self.get_image_with_size(index)
        if image_flag == "gray" or random.random() < self.uncond_p:
            label = -1
        else:
            label = self.get_label(index)

        kwargs = {"index": index}
        return {
            "image": image_tensor,
            "label": label,
            "kwargs": kwargs,
        }


if __name__ == '__main__':
    from processors.arrow_kits import make_arrows
    from processors.image_kits import md5sum_binary

    imagenet_1k_to_arrow("/apdcephfs_nj8/share_301739632/0_public_datasets/imagenet-1k",
                         "__data/imagenet_1k_label.csv",
                         "/apdcephfs_nj8/share_301739632/0_public_datasets/imagenet-1k/ILSVRC2012_validation_ground_truth.txt")
