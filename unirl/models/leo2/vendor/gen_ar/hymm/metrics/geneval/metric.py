# ------------------------------------------------------------------------------
# This script is a modified version of the original GenEval metric script from
# the GenEval repository.
#
# Reference: https://github.com/djghosh13/geneval
# ------------------------------------------------------------------------------

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps

import mmdet
import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
from mmdet.apis import inference_detector, init_detector

zsc.tqdm = lambda it, *args, **kwargs: it

from ..base_metric import BaseMetric
from ...utils.file_utils import safe_file


# Copied from GenEval
THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
TAG2ID = {
    "single_object": 0,
    "two_object": 1,
    "counting": 2,
    "colors": 3,
    "position": 4,
    "color_attr": 5,
}
ID2TAG = {v: k for k, v in TAG2ID.items()}


class ImageCrops(Dataset):
    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        bgcolor = "#999"
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self.transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        # if args.options.get('crop', '1') == '1':
        image = image.crop(box[:4])
        # if args.save:
        #     base_count = len(os.listdir(args.save))
        #     image.save(os.path.join(args.save, f"cropped_{base_count:05}.png"))
        return self.transform(image), 0


class GenEvalMetric(BaseMetric):
    def __init__(self, model_path, dataset_name="GenEval", max_size=1024, device=None):
        super().__init__()
        self.model_path = Path(model_path)
        if device is None:
            if torch.distributed.is_initialized():
                device = torch.distributed.get_rank() % 8
            else:
                device = 0
            self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # We must explicitly set the device to the rank of the process, because modelscope doesn't use pytorch's
        # global rank.
        if not self.model_path.exists():
            raise FileNotFoundError(f"{self.model_path} not found.")
        self._detector = None
        self._clip = None
        self._transform = None
        self._tokenizer = None
        self._color_classifiers = {}
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []

        self.classnames = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "computer mouse", "tv remote", "computer keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        # {} for timestamp
        self.save_file_template = "geneval_{}.csv"

    def load_model(self, logger=None):
        detector = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
        config_path = str(self.model_path / f'configs/mask2former/{detector}.py')
        ckpt_path = str(self.model_path / f"{detector}.pth")
        # config_path = str(self.model_path / "mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py")
        # ckpt_path = str(self.model_path / f"mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth")
        # noinspection PyCallingNonCallable
        self._detector = init_detector(config_path, ckpt_path, device=self.device)

        clip_arch = "ViT-L-14"
        clip_path = str(self.model_path / "ViT-L-14.pt")
        self._clip, _, self._transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained=None, device=self.device)
        jit = torch.jit.load(clip_path, map_location=self.device).eval()
        # torch>2.6时，把 TorchScript 的参数拷贝到 open_clip 模型
        self._clip.load_state_dict(jit.state_dict(), strict=False) 
        self._clip.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_arch)
        if logger is not None:
            logger.info("GenEvalMetric: Detector/clip model loaded.")

    def release_model(self):
        self._detector = None
        self._clip = None
        self._color_classifiers = {}

    @property
    def detector(self):
        if self._detector is None:
            self.load_model()
        return self._detector

    @property
    def clip(self):
        if self._clip is None:
            self.load_model()
        return self._clip

    @staticmethod
    def compute_iou(box_a, box_b):
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        ])
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    @staticmethod
    def relative_position(obj_a, obj_b):
        """Give position of A relative to B, factoring in object dimensions"""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        #
        revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        #
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations

    @staticmethod
    def pt_to_numpy_lst(images: torch.Tensor, scale_up=False) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        if scale_up:
            images = (images * 255).astype(np.uint8)
        images = [x[0] for x in np.split(images, images.shape[0], axis=0)]
        return images

    def color_classification(self, image, bboxes, classname):
        if classname not in self._color_classifiers:
            self._color_classifiers[classname] = zsc.zero_shot_classifier(
                self._clip, self.tokenizer, COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                self.device
            )
        clf = self._color_classifiers[classname]
        dataloader = DataLoader(
            ImageCrops(image, bboxes, self._transform),
            batch_size=16, num_workers=4
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self._clip, clf, dataloader, self.device)
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def evaluate(self, image, objects, metadata):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])[:req['count']]
            if len(found_objects) < req['count']:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if 'color' in req:
                    # Color check
                    colors = self.color_classification(image, found_objects, classname)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found " +
                            f"{colors.count(req['color'])} {req['color']}; and " +
                            ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
                if 'position' in req and matched:
                    # Relative position check
                    expected_rel, target_group = req['position']
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found " +
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        # Check for non-expected objects
        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(objects.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)

    def evaluate_image(self, image, result, metadata, id_):
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        detected = {}
        # Determine bounding boxes to keep
        confidence_threshold = THRESHOLD if metadata['tag'] != "counting" else COUNTING_THRESHOLD
        for index, classname in enumerate(self.classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]    # Threshold
            ordering = ordering[:MAX_OBJECTS].tolist()  # Limit number of detected objects per class
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                ordering = [
                    obj for obj in ordering
                    if NMS_THRESHOLD == 1 or self.compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
                ]
            if not detected[classname]:
                del detected[classname]
        # Evaluate
        is_correct, reason = self.evaluate(image, detected, metadata)
        return {
            'index': id_,
            'correct': is_correct,
            'tag': metadata['tag'],
            'prompt': metadata['prompt'],
        }

    @torch.no_grad()
    def process(self, images, metadata, ids, **kwargs):
        """
        Args:
            images (torch.cuda.Tensor): batch of image tensors with shape (B, 3, H, W) and interval [0, 1]
                or list of image paths.
            metadata (list of dict): each dict contains the metadata for the corresponding image
            **kwargs:
        """

        # torch.Tensor to ndarray
        if isinstance(images[0], torch.Tensor):
            images = self.pt_to_numpy_lst(images, scale_up=True)
        results = inference_detector(self.detector, images)
        scores = []
        for image, result, metadata_i, id_ in zip(images, results, metadata, ids):
            if isinstance(image, str):
                pil_image = ImageOps.exif_transpose(Image.open(image))
            else:
                pil_image = Image.fromarray(image)
            score = self.evaluate_image(pil_image, result, metadata_i, id_)
            scores.append(score)

        self.results.extend(scores)

    def compute_metrics(self, results, save_file=None):
        df = pd.DataFrame(results)
        if save_file:
            df_sorted = df.copy()
            # Convert [3, 10003, 20003, 30003] to [3, 3.1, 3.2, 3.3], making the images of the same prompt successive.
            df_sorted['sub_index'] = df_sorted['index'].apply(lambda x: (x % 10000) + (x // 10000 * 0.1))
            df_sorted.sort_values(by=['sub_index']).to_csv(safe_file(save_file), index=False)
        out_dict = {}
        scores = []

        for tag, group in sorted(df.groupby('tag'), key=lambda x: TAG2ID[x[0]]):
            tag_score = group['correct'].mean()
            print(f"GenEvalMetric: predictions({tag}).shape is", len(group))
            out_dict[tag] = (round(float(tag_score), 6), len(group))
            scores.append(tag_score)

        count = len(df)
        print("GenEvalMetric: predictions.shape is", count)
        out_dict['avg'] = (
            round(float(np.mean(scores)), 6),
            count
        )

        return out_dict
