import cv2
import einops
import numpy as np
import torch
from index_kits import ArrowIndexV2
from PIL import Image
from typing import Dict
from transformers.image_processing_utils import BatchFeature
from torchvision.transforms import transforms
from index_kits import ResolutionGroup

from ..utils.helpers import to_2tuple
from ..utils.import_utils import require_version

require_version("index-kits", "0.5.0", "DefaultImageProcessor")


class DefaultImageProcessor(object):
    def __init__(
        self,
        image_size: int,
        pad_color=(127, 127, 127),
    ):
        self.image_size = to_2tuple(image_size)
        self.pad_color = pad_color

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

    def preprocess(self, images, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        # resize and pad to [self.image_size, self.image_size]
        # then convert from [H, W, 3] to [3, H, W]
        is_single = False
        if isinstance(images, Image.Image):
            images = [images]
            is_single = True

        pixel_values = []
        for image in images:
            image, _ = ArrowIndexV2.resize_and_pad(
                image, self.image_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color
            )
            image = self.transform(image)
            pixel_values.append(image)

        if is_single:
            return pixel_values[0]

        data = {"pixel_values": torch.stack(pixel_values)}
        return BatchFeature(data=data, tensor_type=return_tensors)


SiglipImageProcessor = DefaultImageProcessor


class VAEImageProcessor(object):
    def __init__(
            self,
            base_size: int = 1024,
            reso_step: int = None,
            resolutions: ResolutionGroup = None,
            vae_trans_type: str = "-11",
    ):
        self.base_size = base_size
        self.reso_step = reso_step
        if reso_step is None:
            reso_step = base_size // 16
        self.resolutions = resolutions

        if resolutions is None:
            self.resolutions = ResolutionGroup(base_size=base_size, step=reso_step)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        if vae_trans_type == "-11":
            self.normalizer = transforms.Compose([
                    transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
                ])
        elif vae_trans_type == "01":
            self.normalizer = transforms.Compose([])
        else:
            raise ValueError("Invalid vae_trans_type, should be '-11' or '01'.")

    def normalize(self, images):
        if isinstance(images, Image.Image):
            outputs = self.normalizer(transforms.ToTensor()(images))
        elif isinstance(images, torch.Tensor):
            assert images.ndim == 3 or images.ndim == 4, f"images should be 3D or 4D tensor, got {images.ndim}"
            outputs = self.normalizer(images)
        elif isinstance(images, list):
            outputs = []
            for image in images:
                if isinstance(image, torch.Tensor):
                    assert image.ndim == 3 or image.ndim == 4, f"image should be 3D or 4D tensor, got {image.ndim}"
                    outputs.append(self.normalizer(image))
                elif isinstance(image, list):
                    inner_outputs = []
                    for im in image:
                        assert im.ndim == 3 or im.ndim == 4, f"image should be 3D or 4D tensor, got {im.ndim}"
                        inner_outputs.append(self.normalizer(im))
                    outputs.append(inner_outputs)
                else:
                    raise ValueError(f"image should be tensor or list of tensor, got {type(image)}")
        else:
            raise ValueError(f"images should be tensor or list, got {type(images)}")
        return outputs

    def preprocess(self, images, return_tensors: str = "pt", **kwargs) -> Dict:
        is_single = False
        if isinstance(images, Image.Image):
            images = [images]
            is_single = True

        pixel_values = []
        for image in images:
            origin_size = image.size    # (ow, oh)
            target_size = self.resolutions.get_target_size(*origin_size)    # (tw, th)
            image, _ = ArrowIndexV2.resize_and_crop(
                image, target_size, crop_type="center", resample=Image.Resampling.BICUBIC
            )
            image = self.transform(image)
            pixel_values.append(image)

        if is_single:
            return pixel_values[0]

        if all(pv.shape == pixel_values[0].shape for pv in pixel_values):
            return {"pixel_values": torch.stack(pixel_values)}
        else:
            return {"pixel_values": pixel_values}


class FaceImageProcessor(object):
    def __init__(self):
        self._preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.tensor_to_pil_image = transforms.Compose(
            [
                transforms.Normalize([-1], [2]),
                transforms.ToPILImage(),
            ]
        )

    def preprocess(self, images):
        if isinstance(images, Image.Image):
            return self._preprocess(images)
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                return images
            elif images.ndim == 4 and images.shape[0] == 1:
                return images[0]
            else:
                raise ValueError(f"images can be 3D or 4D tensor with shape [1, c, h, w], got {images.shape}")
        else:
            raise ValueError(f"images should be PIL Image or torch Tensor, got {type(images)}")

    @staticmethod
    def get_face_info(face_analysis, face_image_torch):
        """
            get the face info from the face image
        """
        face_image_np = face_image_torch.cpu().numpy()
        face_image_np = einops.rearrange(face_image_np, 'c h w -> h w c')
        face_image_np = (face_image_np + 1) * 127.5
        face_image_np = face_image_np.astype(np.uint8)  # h w c uint8 range: 0, 255
        face_analysis_input = cv2.cvtColor(face_image_np, cv2.COLOR_RGB2BGR)
        face_info = face_analysis.get(face_analysis_input)
        return face_info

    @staticmethod
    def resize_and_pad(input_image, max_side=(256, 256), size=None, pad_to_max_side=True, base_pixel_number=1):
        """
        resize torch tensor [n, c, h, w] range(-1, 1), resize long side to max_side, pad short side to max_side
        Args:
            input_image (torch tensor): [n, c, h, w] float number
            max_side (list of int tw, th, optional): For tuple [ (tw, sw), (th, sh)], resize side with smaller t/s to corresponding axis, pad larger ratio axis to max_side
            size: resize to size
            pad_to_max_side (bool, optional): pad short side to max_side
            base_pixel_number (int, optional): resize to base_pixel_number
        Returns:
            torch tensor: [n, c, max_side, w* (max_side/h)] or [n, c, h * (max_side/w), max_side] -1, 1

        """
        tw, th = max_side
        n, c, h, w = input_image.shape
        if size is not None:
            h_resize_new, w_resize_new = size
        else:
            # Calculate the ratio to resize the image so the larger side is max_side
            ratio = min(tw / w, th / h)
            w, h = round(ratio * w), round(ratio * h)
            w_resize_new = (w // base_pixel_number) * base_pixel_number
            h_resize_new = (h // base_pixel_number) * base_pixel_number

        input_image = torch.nn.functional.interpolate(input_image, size=(h_resize_new, w_resize_new), mode='bicubic')
        input_image = input_image.clamp(-1, 1)

        if pad_to_max_side:
            res = torch.zeros([n, c, th, tw], dtype=torch.float32)
            offset_x = (tw - w_resize_new) // 2
            offset_y = (th - h_resize_new) // 2
            res[:, :, offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = input_image
            input_image = res
        return input_image

    def crop_face_image_torch(self, face_analysis, face_image_torch, logger=None, return_face=False):
        """
            crop the face area from image torch tensor, resize to max_side=256, and return the face image torch tensor

        Args:
            face_analysis: instance of the face analysis
            face_image_torch (torch tensor): [3, sh, sw] float32 range: -1, 1
            logger:
        Returns:
            torch tensor: [3, tw, th] float32 range: -1, 1
        """
        face_info = self.get_face_info(face_analysis, face_image_torch)

        # Handle multiple faces, small face, and negative face_bbox and no face

        if len(face_info) >= 1:
            face_info = sorted(
                face_info,
                key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
            )[-1]  # only use the maximum face
            face_embedding = face_info.get('embedding', None)
            if isinstance(face_embedding, np.ndarray):
                face_embedding = torch.from_numpy(face_embedding)
        else:
            logger.info(f"WARNING: no face detected, skip face crop")
            face_embedding = None

        if return_face:
            face_bbox = face_info['bbox']
            # if area of face is too small, skip
            # TODO: 10% probability to get negative face_bbox coordinates; can be improved later
            if face_bbox[2] - face_bbox[0] < 10 or face_bbox[3] - face_bbox[1] < 10:
                face_image_crop_torch = face_image_torch
            else:
                face_image_crop_torch = face_image_torch[:, int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
            try:
                face_image_crop_resize_torch = self.resize_and_pad(
                    face_image_crop_torch[None], max_side=(512, 512), pad_to_max_side=True)[0]
            except Exception as e:
                logger.info(
                    f"Error: input face_image_torch shape: {face_image_torch.shape}; "
                    f"face_image_crop_torch shape: {face_image_crop_torch.shape}")
                raise e

            face_image = self.tensor_to_pil_image(face_image_crop_resize_torch)
            return face_embedding, face_image

        return face_embedding

    def extract_face(self, face_analysis, processed_image, logger, return_face=False):
        return self.crop_face_image_torch(
            face_analysis,
            processed_image,
            logger=logger,
            return_face=return_face,
        )
