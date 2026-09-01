import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from loguru import logger

from hymm.data_kits.datasets.coco import COCOValDataset
from hymm.data_kits.imagenet import ImageNetDataset
from hymm.metrics.fid.metric import FIDMetric
from hymm.constants import FID_INCEPTION_PATH
from hymm.core.global_vars import get_nccl_timeout


def init_distributed_mode(args):
    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    logger.info("| distributed worker initialized as global_rank {}/{}, local_rank {}".format(
        args.rank, args.world_size, args.local_rank
    ))
    dist.init_process_group(backend=args.dist_backend, timeout=get_nccl_timeout())
    dist.barrier()


def main(args):
    init_distributed_mode(args)
    device = torch.device("cuda")

    seed = args.rank
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_path = Path(args.save_path)

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )
    if args.dataset == "coco":
        dataset = COCOValDataset(args.data, transform)
    elif args.dataset == "imagenet":
        def imagenet_collate_fn(batch):
            batch_size = len(batch)

            images = []
            image_ids = []

            for i in range(batch_size):
                images.append(batch[i]["image"])
                image_ids.append(batch[i]["label"])

            return {
                "images": torch.stack(images, dim=0),
                "image_ids": image_ids,
            }

        dataset = ImageNetDataset(
            image_size=256,
            index_file=args.data,
            crop_type="center",
            logger=logger,
        )
        dataset.collate_fn = imagenet_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=dataset_sampler,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
        batch_size=args.batch_size,
        num_workers=0,
    )

    fid_metric = FIDMetric(dims=2048, inception_path=FID_INCEPTION_PATH, img_save_path=None, target_path=None)
    torch.distributed.barrier()

    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            if args.rank == 0:
                print(batch_id)
            fid_metric.process(batch["images"].to(device))
            torch.distributed.barrier()

    fid_results = fid_metric.all_gather_results()
    if args.rank == 0:
        mu, sigma = fid_metric.compute_stats(fid_results)
        print(mu.shape, sigma.shape)
        torch.save({"mu": mu, "sigma": sigma}, save_path)
    torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Make COCO/Imagenet FID Stats", add_help=False)
    parser.add_argument("--dataset", type=str, required=True, choices=["coco", "imagenet"], help="dataset name")
    parser.add_argument("--data", type=str, required=True, help="data path")
    parser.add_argument("--batch-size", type=int, required=True, help="batch size per GPU")
    parser.add_argument("--save-path", type=str, required=True, help="save path")

    args = parser.parse_args()

    main(args)

"""
PYTHONPATH=./ torchrun --master_addr=127.0.0.1 --master_port=2333 --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  ./hymm/metrics/fid/make_fid_stats.py \
  --dataset coco \
  --data /apdcephfs/ckczzjzhang/data/COCO/val2014 \
  --batch-size 200 \
  --save-path ./coco_val_fid_stats.pt
  
PYTHONPATH=./ torchrun --master_addr=127.0.0.1 --master_port=2333 --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  ./hymm/metrics/fid/make_fid_stats.py \
  --dataset imagenet \
  --data __data/dataset_image/imagenet_1k_val.json \
  --batch-size 200 \
  --save-path ./imagenet_val_fid_stats.pt
"""
