from typing import Any, Dict, List, Union, Optional

import numpy as np
from pycocotools import mask as coco_mask

from hymm.data_kits.grounding_data_processors.base_processor import BaseDataProcessor
from index_kits import ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2


def rle_to_mask(rle_dict: Dict[str, Any], image_height=None, image_width=None) -> np.ndarray:
    """Convert COCO-format RLE encoding to a binary mask"""
    if 'size' in rle_dict:
        h, w = rle_dict['size']
    else:
        assert image_width is not None and image_height is not None
        h, w = image_height, image_width

    if isinstance(rle_dict['counts'], list):
        rle_obj = coco_mask.frPyObjects(rle_dict, h, w)
    else:
        rle_obj = {'counts': rle_dict['counts'], 'size': [h, w]}

    mask = coco_mask.decode(rle_obj).astype(np.uint8)
    return mask


class CocoDataProcessor(BaseDataProcessor):
    """Processor for COCO format data"""
    def __init__(self,
                 index_manager: Union[ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2],
                 mask_to_img_mode: str = "RGB",
                 sample_mask_rate: float = 0.9,
                 n_select: Optional[int] = None):
        """
        Args:
            index_manager: the index manager
            mask_to_img_mode: the mode of the mask to image, "RGB" or "L"
            sample_mask_rate: the rate of sampling masks
            n_select: int, the number of masks to sample, default is None
                - If None, sample masks randomly
                - If not None, sample `n_select` masks every time
        """
        super().__init__(index_manager, mask_to_img_mode)
        self.sample_mask_rate = sample_mask_rate
        self.n_select = n_select
    
    def rle_to_mask(self, rle_dict: Dict[str, Any]) -> np.ndarray:
        """Convert COCO-format RLE encoding to a binary mask"""
        return rle_to_mask(rle_dict, self.image_height, self.image_width)

    def polygon_to_mask(self, polygons: List[List[float]]) -> np.ndarray:
        """Convert polygons to a binary mask"""
        rle = coco_mask.frPyObjects(polygons, self.image_height, self.image_width)
        
        if isinstance(rle, list):
            merged_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            for r in rle:
                mask = coco_mask.decode(r).astype(bool)
                merged_mask |= mask
            return merged_mask.astype(np.uint8)
        else:
            return coco_mask.decode(rle).astype(np.uint8)
    
    def get_data_from_arrow(self, index: int) -> Any:
        """Get data from coco arrow file"""
        segmentation = self.index_manager.get_attribute(index, "segmentation")
        segmentation = self.string_to_list(segmentation)
        bboxes = self.index_manager.get_attribute(index, "bbox")
        category_names = self.index_manager.get_attribute(index, "category_names")
        is_crowd = self.index_manager.get_attribute(index, "is_crowd")

        self.data = {
            "segmentation": segmentation,
            "bbox": bboxes,
            "category_names": category_names,
            "is_crowd": is_crowd,
        }
        return self.data

    def merge_masks_and_boxes_by_caption(
        self,
        data: Dict[str, List[Any]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Merge masks and boxes with the same caption for COCO data"""
        merged_masks = {}
        
        for i, caption in enumerate(data['category_names']):
            # Check if the box is valid
            if data['bbox'][i][2] <= data['bbox'][i][0] or data['bbox'][i][3] <= data['bbox'][i][1]:
                continue

            # Generate mask based on its type (RLE or polygon)
            if data['is_crowd'][i] == 1:
                mask = self.rle_to_mask(data['segmentation'][i])
            else:
                mask = self.polygon_to_mask(data['segmentation'][i])

            box_mask = self.box_to_mask(data['bbox'][i])
            
            # Merge masks with the same caption
            if caption not in merged_masks:
                merged_masks[caption] = {
                    'mask': mask,
                    'box_mask': box_mask,
                }
            else:
                merged_masks[caption]['mask'] |= mask
                merged_masks[caption]['box_mask'] |= box_mask
        
        return merged_masks

    def __call__(self, 
                 index: int,
                 image_height: int,
                 image_width: int,
                 sample_type: str = "random",
                 return_caption: bool = False) -> Dict[str, Any]:
        """Randomly sample masks from arrow file

        Args:
            index: int, the index of the image in the arrow file
            image_height: int, the height of the image
            image_width: int, the width of the image
            sample_type: str, the type of sample to use, "mask", "bbox" or "random"
                - "mask": sample masks
                - "bbox": sample bbox-type masks
                - "random": sample masks or bbox-type masks randomly

        Returns:
            Dict[str, Any]: A dictionary containing the sampled masks and their captions.
            The dictionary has the following keys:
                - 'system_prompt': str, the system prompt
                - 'instruction': str, the instruction
                - 'mask': PIL.Image.Image, the sampled mask, default shape is (H, W, 3)
        """
        try:
            self.set_image_size(image_height, image_width)
            grounding_data = self.get_data_from_arrow(index)
            captions = self.index_manager.get_attribute(index, "captions")
            caption = captions[np.random.randint(0, len(captions))]
            merged_masks = self.merge_masks_and_boxes_by_caption(grounding_data)
            assert len(merged_masks) > 0, "No masks found, set a dummy caption instead."
        except:
            # If no masks are found, set a dummy caption
            caption = " "
            merged_masks = {}
            merged_masks[caption] = {
                'mask': np.zeros((self.image_height, self.image_width), dtype=np.uint8),
                'box_mask': np.zeros((self.image_height, self.image_width), dtype=np.uint8),
            }

        # Randomly sample masks or bbox-type masks
        # ---- caption: str, the caption of the sampled masks, e.g. "a car, a tree"
        # ---- mask: np.ndarray, the sampled mask
        # ---- box_mask: np.ndarray, the box mask of the sampled mask
        sampled_mask_info = self.random_sample_masks(merged_masks, self.n_select)

        # Randomly sample masks or bbox-type masks
        if sample_type == "random":
            if np.random.rand() < self.sample_mask_rate:
                sample_type = "mask"
            else:
                sample_type = "bbox"

        if sample_type == "mask":
            mask = sampled_mask_info['mask']
            instruction = self.mask_instruction[np.random.randint(0, len(self.mask_instruction))]
        elif sample_type == "bbox":
            mask = sampled_mask_info['box_mask']
            instruction = self.bbox_instruction[np.random.randint(0, len(self.bbox_instruction))]
        else:
            raise ValueError(f"sample_type is expected to be 'mask' or 'bbox', but got {sample_type}.")
        
        result = {
            "system_prompt": instruction,
            "instruction": sampled_mask_info['caption'],
            "mask": self.mask_to_img(mask),
        }
        if return_caption:
            result['caption'] = caption

        return result
