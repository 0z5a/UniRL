from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from index_kits import ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2

from hymm.data_kits.grounding_data_processors.base_processor import BaseDataProcessor


class GritDataProcessor(BaseDataProcessor):
    """Processor for GRIT format data"""

    def __init__(self,
                 index_manager: Union[ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2],
                 mask_to_img_mode: str = "RGB",
                 box_score_threshold: float = 0.65,
                 n_select: Optional[int] = None):
        super().__init__(index_manager, mask_to_img_mode)
        self.box_score_threshold = box_score_threshold
        self.n_select = n_select
    
    def ref_exps_or_noun_chunks_to_boxes(
        self,
        ref_exps_or_noun_chunks: List[Tuple[int, int, float, float, float, float, float]],
        captions: str,
        clamp: bool = True
    ) -> List[Dict[str, Any]]:
        """Convert ref_exps or noun_chunks to normalized bounding box coordinates"""
        boxes = []
        
        for item in ref_exps_or_noun_chunks:
            if len(item) != 7:
                raise ValueError(f"Invalid ref_exps_or_noun_chunks({item}), expected 7 elements")
                
            phrase_s, phrase_e, x1_n, y1_n, x2_n, y2_n, score = item
            
            x1 = int(round(x1_n * self.image_width))
            y1 = int(round(y1_n * self.image_height))
            x2 = int(round(x2_n * self.image_width))
            y2 = int(round(y2_n * self.image_height))
            
            if clamp:
                x1 = max(0, min(x1, self.image_width - 1))
                y1 = max(0, min(y1, self.image_height - 1))
                x2 = max(0, min(x2, self.image_width - 1))
                y2 = max(0, min(y2, self.image_height - 1))
            
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Check if the box is valid
            if x1 == x2 or y1 == y2:
                continue

            # Extract caption of the current box from captions
            caption = captions[int(phrase_s):int(phrase_e)]

            if score < self.box_score_threshold:
                continue
            
            boxes.append({
                "bbox": (x1, y1, x2, y2),
                "caption": caption,
                "score": score,
            })
            
        return boxes

    def get_data_from_arrow(self, index: int, grounding_caption_type: str = "ref_exps") -> Any:
        """Get data from grit arrow file
        
        Args:
            index: int, the index of the image in the arrow file
            grounding_caption_type: str, the type of grounding caption to use, "ref_exps" or "noun_chunks" or "random"
                - "ref_exps": use reference expressions
                - "noun_chunks": use noun chunks
                - "random": randomly choose ref_exps or noun_chunks
        """
        ref_exps = self.index_manager.get_attribute(index, "ref_exps")
        noun_chunks = self.index_manager.get_attribute(index, "noun_chunks")
        caption = self.index_manager.get_attribute(index, "caption")

        assert grounding_caption_type in ["ref_exps", "noun_chunks", "random"], f"grounding_caption_type is expected to be 'ref_exps' or 'noun_chunks' or 'random', but got {grounding_caption_type}."

        # Randomly choose ref_exps or noun_chunks
        if grounding_caption_type == "ref_exps":
            ref_exps_or_noun_chunks = ref_exps
        elif grounding_caption_type == "noun_chunks":
            ref_exps_or_noun_chunks = noun_chunks
        else:
            ref_exps_or_noun_chunks = np.random.choice([ref_exps, noun_chunks])

        self.data = self.ref_exps_or_noun_chunks_to_boxes(
            ref_exps_or_noun_chunks=ref_exps_or_noun_chunks,
            captions=caption,
        )
        return self.data
       
    def merge_masks_and_boxes_by_caption(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge masks and boxes with the same caption for GRIT data"""
        
        merged_masks = {}
        
        for box_info in data:
            caption = box_info['caption']
            score = box_info['score']
            bbox = box_info['bbox']
            
            if caption not in merged_masks:
                merged_masks[caption] = {
                    'box_mask': self.box_to_mask(bbox),
                    'score': score,
                }
            else:
                merged_masks[caption]['box_mask'] |= self.box_to_mask(bbox)
                
        return merged_masks

    def __call__(self, 
                 index: int,
                 image_height: int,
                 image_width: int,
                 sample_type: str = "random",
                 grounding_caption_type: str = "ref_exps",
                 return_caption: bool = False) -> Dict[str, Any]:
        """Randomly sample masks from arrow file

        Args:
            index: int, the index of the image in the arrow file
            image_height: int, the height of the image
            image_width: int, the width of the image
            sample_type: str, the type of sample to use, "bbox" or "random"
            grounding_caption_type: str, the type of grounding caption to use, "ref_exps" or "noun_chunks" or "random"
                - "ref_exps": use reference expressions
                - "noun_chunks": use noun chunks
                - "random": randomly choose ref_exps or noun_chunks

        Returns:
            Dict[str, Any]: A dictionary containing the sampled masks and their captions.
            The dictionary has the following keys:
                - 'system_prompt': str, the system prompt
                - 'instruction': str, the instruction
                - 'mask': PIL.Image.Image, the sampled mask (default in RGB mode), default shape is (H, W, 3)
        """
        try:
            self.set_image_size(image_height, image_width)
            grounding_data = self.get_data_from_arrow(index, grounding_caption_type)
            caption = self.index_manager.get_attribute(index, "caption")
            merged_masks = self.merge_masks_and_boxes_by_caption(grounding_data)
            assert len(merged_masks) > 0, "No masks found, set a dummy caption instead."
        except:
            # If no masks are found, set a dummy caption
            caption = " "
            merged_masks = {}
            merged_masks[caption] = {
                'box_mask': np.zeros((self.image_height, self.image_width), dtype=np.uint8),
            }

        # Randomly sample masks (maybe) or bbox-type masks
        # ---- caption: str, the caption of the sampled masks, e.g. "a car, a tree"
        # ---- mask: np.ndarray, the sampled mask
        # ---- box_mask: np.ndarray, the box mask of the sampled mask
        sampled_mask_info = self.random_sample_masks(merged_masks, self.n_select)

        # GRIT data only has bbox-type masks
        if sample_type == "random":
            sample_type = "bbox"

        if sample_type == "bbox":
            mask = sampled_mask_info['box_mask']
            instruction = self.bbox_instruction[np.random.randint(0, len(self.bbox_instruction))]
        else:
            raise ValueError(f"`sample_type` is expected to be 'bbox' for GRIT data, but got {sample_type}.")
        
        result = {
            "system_prompt": instruction,
            "instruction": sampled_mask_info['caption'],
            "mask": self.mask_to_img(mask),
        }
        if return_caption:
            result['caption'] = caption

        return result
   