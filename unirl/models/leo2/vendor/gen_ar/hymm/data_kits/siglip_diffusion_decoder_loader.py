import torch

from hymm.data_kits.image_loader import ImageArrowStream
from hymm.constants import VISION_ENCODER_META_INFO

from transformers.models.siglip2.image_processing_siglip2_fast import Siglip2ImageProcessorFast

class SiglipDiffusionDecoderArrowStream(ImageArrowStream):
    def __post_init__(self):
        self.vision_encoder_processor = Siglip2ImageProcessorFast.from_pretrained(VISION_ENCODER_META_INFO[self.args.vision_encoder_type]["path"])
        self.vision_encoder_max_num_patches = self.args.vision_encoder_max_num_patches

    def __getitem__(self, index):
        """
        Get image and text from a given index

        Args:
            index (int): Index of the dataset

        Returns:
            image (torch.FloatTensor): Image tensor with shape (3, H, W)
            text (str): Original text
            kwargs (dict): Additional information
                origin_size (torch.LongTensor): Original size of the image (W, H)
                target_size (torch.LongTensor): Target size of the image (W, H)
                crop_coords_xy (torch.LongTensor): Crop coordinates (x, y)
                index (torch.LongTensor): Index of the dataset
        """

        # Get image
        image_tensor, pil_image, kwargs, image_flag = self.get_image_with_size(index, image_key=self.image_key, shadow=self.image_key_shadow)
        
        vision_encoder_inputs = self.vision_encoder_processor(pil_image, max_num_patches=self.vision_encoder_max_num_patches)
        # 1 x max_num_patches x 768
        pixel_values=vision_encoder_inputs["pixel_values"]
        pixel_values=pixel_values.squeeze(0)
        # 1 x max_num_patches
        pixel_attention_mask=vision_encoder_inputs["pixel_attention_mask"]
        pixel_attention_mask=pixel_attention_mask.squeeze(0)
        # 1 x 2 (h, w) 
        spatial_shapes=vision_encoder_inputs["spatial_shapes"]
        spatial_shapes=spatial_shapes.squeeze(0)
        # h x w 是pixel_attention_mask中1的个数
        # h, w = spatial_shapes[0], spatial_shapes[1]

        # Prepare kwargs
        kwargs["index"] = index

        ret = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
            "image_tensor": image_tensor,
            "kwargs": {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        }

        return ret
