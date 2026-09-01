import os
import os.path as osp
import io
import glob
import argparse
import numpy as np
import pyarrow as pa
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from PIL import Image
import torch
from .torch_utils import instantiate_from_config
from .file_utils import load_yaml


def get_table(arrow_file):
    return pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, "r")).read_all()


class OneImageArrow(Dataset):
    def __init__(self, arrow_file, img_key="image", img_size=256):
        self.table = get_table(arrow_file)
        self.img_key = img_key
        self.image_size = (img_size, img_size)
        self.transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor(), Normalize(0.5, 0.5)])

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        try:
            img = self.table[self.img_key][idx].as_py()
            image_bytes = io.BytesIO(img)
            image_bytes.seek(0)
            pil_image = Image.open(image_bytes).convert("RGB")
        except Exception as e:
            print("Error occurs at idx: {} exception {}".format(idx, str(e)))
            pil_image = Image.new("RGB", (self.image_size[0], self.image_size[1]), (128, 128, 128))
        img_tensor = self.transform(pil_image)
        data_dict = {"image": img_tensor, "id": idx}
        return data_dict


def OneImageArrowLoader(arrow_file, img_key="image", img_size=256, batch_size=1, shuffle=False, num_workers=4):
    dataset = OneImageArrow(arrow_file, img_key, img_size)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False
    )


def build_model_from_config(config, ckpt):
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt)
    return model


def get_table(arrow_file):
    return pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, "r")).read_all()


def extract_one_arrow_file(model, arrow_file, batch_size=100, k="image", device_id=0):
    loader = OneImageArrowLoader(arrow_file, batch_size=batch_size)
    total_batch = len(loader)
    token_tensor_list = []
    for i, data_dict in enumerate(loader):
        idx = data_dict["id"]
        final_tensor = data_dict["image"].to(device_id)
        logger.info(f"Rank {args.rank} processing batch {i}/{total_batch}")
        with torch.no_grad():
            tokens = model.vq_encode(final_tensor)
        logger.info(f"tokens shape: {tokens.shape}")
        token_npy = tokens.long().cpu().numpy()
        for tidx in range(token_npy.shape[0]):
            token_tensor_list.append({"image_token": token_npy[tidx].flatten()})
    return token_tensor_list


def get_output_file_path(output_folder, arrow_file):
    base_name = os.path.basename(arrow_file)
    base_folder = osp.basename(osp.dirname(arrow_file))
    second_folder = osp.join(output_folder, base_folder)
    if not osp.exists(second_folder):
        os.makedirs(second_folder)
    return os.path.join(second_folder, base_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_yaml", type=str, default="mimo/configs/autoencoder/ldm-vq-f8.yaml")
    parser.add_argument("--ckpt", type=str, default="/apdcephfs_sh7/share_301124792/milesjyang/workspace/vae/vq-f8/model_pure.ckpt")
    parser.add_argument(
        "--input_arrow_list",
        type=str,
        default="/apdcephfs_sh7/share_301124792/milesjyang/text_img_dataset/images/text2image_*/*.arrow",
    )
    parser.add_argument(
        "--output_root_folder",
        type=str,
        default="/apdcephfs_sh7/share_301124792/milesjyang/text_img_dataset/images_token_vq_f8/",
    )
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args, unkown_args = parser.parse_known_args()
    logger.info(args)
    device_id = args.rank % torch.cuda.device_count()
    # build model
    model_cfg = load_yaml(args.model_yaml)
    logger.info(model_cfg)
    vae_model = build_model_from_config(model_cfg, args.ckpt)
    vae_model.eval()
    logger.info(vae_model)
    vae_model.to(device_id)
    # split arrow files
    total_arrow_list = sorted(glob.glob(args.input_arrow_list))
    split_arrow_list = np.array_split(total_arrow_list, args.world_size)
    logger.info(f"Total {len(total_arrow_list)} arrows, split into {len(split_arrow_list)} splits")
    logger.info(f"Current rank {args.rank}, handle {len(split_arrow_list[args.rank])} ranks")
    for idx, arrow_file in enumerate(split_arrow_list[args.rank]):
        logger.info(f"Current rank {args.rank}, handle {idx} arrow file {arrow_file}")
        result_list_dict = extract_one_arrow_file(
            vae_model, arrow_file, batch_size=args.batch_size, device_id=device_id
        )
        output_file = get_output_file_path(args.output_root_folder, arrow_file)
        df = pd.DataFrame(result_list_dict)
        pa_table = pa.Table.from_pandas(df)
        with pa.OSFile(output_file, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, pa_table.schema) as writer:
                writer.write_table(pa_table)
        logger.info(f"Current rank {args.rank}, handle {idx} Finish writing {output_file}")
