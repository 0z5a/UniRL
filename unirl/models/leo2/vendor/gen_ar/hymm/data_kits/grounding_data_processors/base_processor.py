from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from index_kits import ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2


GROUNDING_MASK_INSTRUCTION = [
    "Identify and mask the objects as specified in the prompt.",
    "Generate masks for the elements based on the given description.",
    "Create masks for the subjects following the provided instructions.",
    "Outline the components using the details from the prompt.",
    "Segment the items according to the description below.",
    "Produce masks for the objects per the written guide.",
    "Highlight the elements based on the text prompt.",
    "Mark the subjects following the outlined instructions.",
    "Define the areas of the objects as described here.",
    "Generate segmentation masks using the provided details.",
    "Isolate the components according to the given scenario.",
    "Create masks for the described elements in the prompt.",
    "Identify and outline the subjects based on the instructions.",
    "Produce segmentation masks for the objects as specified.",
    "Generate masks for the described items using the given guidelines.",
]

GROUNDING_BBOX_INSTRUCTION = [
    "Identify and detect the bounding box of the objects as specified in the prompt.",
    "Generate rectangular boxes to locate elements based on the given description.",
    "Create bounding boxes for the subjects following the provided instructions.",
    "Draw boxes around the components using the details from the prompt.",
    "Detect and box the items according to the description below.",
    "Produce rectangular regions for the objects per the written guide.",
    "Highlight objects with bounding boxes based on the text prompt.",
    "Mark the subjects with rectangular boxes following the outlined instructions.",
    "Define the bounding regions of the objects as described here.",
    "Generate detection boxes using the provided details.",
    "Isolate objects with rectangular boundaries according to the given scenario.",
    "Create bounding boxes for the described elements in the prompt.",
    "Identify and outline objects with rectangular regions based on the instructions.",
    "Produce detection boxes for the objects as specified.",
    "Generate bounding boxes for the described items using the given guidelines.",
]


class BaseDataProcessor(ABC):
    """Base class for grounding data processing"""

    def __init__(self,
                 index_manager: Union[ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2],
                 mask_to_img_mode: str = "RGB"):
        self.index_manager = index_manager
        assert mask_to_img_mode in ["RGB", "L"], f"mask_to_img_mode is expected to be 'RGB' or 'L', but got {mask_to_img_mode}."
        self.mask_to_img_mode = mask_to_img_mode

        # Set instruction for grounding task including mask and bbox
        self.set_instruction()
    
    def get_data_from_arrow(self, index: int) -> Any:
        """Get data from arrow dataset"""
        self.data = None

    def set_image_size(self, image_height: int, image_width: int):
        self.image_height = image_height
        self.image_width = image_width
    
    def box_to_mask(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert box coordinates to binary mask

        Args:
            box: Tuple of (x1, y1, x2, y2).
        Returns:
            np.ndarray: Binary mask of the box.
        """
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        box = [int(b) for b in box]
        mask[box[1]:box[3], box[0]:box[2]] = 1
        return mask
    
    def mask_to_img(self, mask, save_path=None):
        assert self.mask_to_img_mode in ["RGB", "L"], f"to_mode is expected to be 'RGB' or 'L', but got {self.mask_to_img_mode}."
        if self.mask_to_img_mode == "RGB":
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)

        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode=self.mask_to_img_mode)
        if save_path:
            mask_image.save(save_path)
        return mask_image
    
    @staticmethod
    def mask_apply_to_image(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
        """Apply red transparent mask to image
        
        Args:
            image: PIL Image in any mode
            mask: PIL Image mask in any mode
            alpha: transparency of the red mask (0-1)
            
        Returns:
            PIL Image with red mask overlay
        """
        # Convert image to RGBA and mask to grayscale
        image = image.convert("RGBA")
        mask = mask.convert("L")
        
        # Ensure mask has same size as image
        assert image.size == mask.size, f"image.size={image.size} != mask.size={mask.size}"
        
        # Create red transparent mask using numpy operations
        mask_array = np.array(mask)
        rgba = np.zeros((*mask.size[::-1], 4), dtype=np.uint8)
        rgba[..., 0] = 255  # Red channel
        rgba[..., 3] = np.where(mask_array > 128, int(255 * alpha), 0)  # Alpha channel
        
        # Create red mask image and blend
        red_mask = Image.fromarray(rgba, mode='RGBA')
        blended = Image.alpha_composite(image, red_mask)
        # rgba mode to rgb mode
        blended = blended.convert("RGB")
        return blended
    
    @abstractmethod
    def merge_masks_and_boxes_by_caption(self, data: Any) -> Dict[str, Dict[str, Any]]:
        """Merge masks (and boxes-type masks) with the same caption"""
        pass

    @staticmethod
    def string_to_list(string: str) -> List[Any]:
        """Convert string to list"""
        return eval(string)

    def set_instruction(self):
        """Set the instruction for the grounding task"""
        self.mask_instruction = GROUNDING_MASK_INSTRUCTION
        self.bbox_instruction = GROUNDING_BBOX_INSTRUCTION
    
    @staticmethod
    def random_sample_masks(masks: Dict[str, Any], n_select: Optional[int] = None) -> Dict[str, Any]:
        """Randomly sample masks from a dictionary of masks

        Args:
            masks: A dictionary of masks.
            The dictionary has the following keys:
                - 'caption': str, the caption of the masks
                - (maybe) 'mask': np.ndarray, the mask
                - (maybe) 'box_mask': np.ndarray, the box mask of the mask

        Returns:
            Dict[str, Any]: A dictionary containing the sampled masks and their captions.
            The dictionary has the following keys:
                - 'caption': str, the joined caption of the sampled masks, e.g. "a car, a tree"
                - (maybe) 'mask': np.ndarray, the sampled mask, shape is (H, W)
                - (maybe) 'box_mask': np.ndarray, the box mask of the sampled mask, shape is (H, W)
        """

        # Sample masks
        # Determine number of masks to select (between 1 and total available)
        if n_select is None:
            n_select = np.random.randint(1, len(masks) + 1)
        else:
            assert n_select > 0, f"n_select is expected to be greater than 0, but got {n_select}."
            n_select = min(n_select, len(masks))

        # Randomly select n_select masks
        selected_items = list(masks.items())
        selected_indices = np.random.choice(len(selected_items), size=n_select, replace=False)
        
        # Initialize result with first selected mask
        first_caption, first_mask_info = selected_items[selected_indices[0]]
        result = {
            'caption': first_caption,
            **{k: v.copy() if isinstance(v, np.ndarray) else v 
               for k, v in first_mask_info.items()}
        }
        
        # Merge remaining selected masks
        for idx in selected_indices[1:]:
            caption, mask_info = selected_items[idx]
            # Update caption
            result['caption'] += ', ' + caption 
            
            # Merge masks
            for key, value in mask_info.items():
                if 'mask' in key:
                    result[key] |= value

        return result
    
    def __call__(self, 
                 index: int,
                 image_height: int,
                 image_width: int,
                 mask_type: str = "mask") -> Dict[str, Any]:
        """Randomly sample masks from arrow file"""
        pass

    def visualize_mask_with_caption(self, image: Image.Image, mask: Image.Image, caption: str, alpha: float = 0.5) -> Image.Image:
        """Visualize mask on the image with caption.
        
        Args:
            image: PIL Image in RGB format
            mask: PIL Image in RGB format representing the mask
            caption: Text to display above the image
            alpha: Transparency of the mask overlay (0.0 to 1.0)
            
        Returns:
            PIL Image with mask overlay and caption
        """
        # Convert images to RGBA
        image = image.convert("RGBA")
        
        # Convert mask to red with transparency
        mask = mask.convert("L")  # Convert to grayscale
        red_mask = Image.new("RGBA", mask.size, (255, 0, 0, 0))  # Create transparent red image
        pixels = mask.load()  # Get pixel access
        red_pixels = red_mask.load()
        
        # Set alpha channel based on mask values
        for i in range(mask.width):
            for j in range(mask.height):
                if pixels[i, j] > 128:  # If pixel is white-ish
                    red_pixels[i, j] = (255, 0, 0, int(255 * alpha))  # Red with alpha
                else:
                    red_pixels[i, j] = (255, 0, 0, 0)  # Transparent
        
        # Ensure mask has same size as image
        if image.size != red_mask.size:
            red_mask = red_mask.resize(image.size)
        
        # Blend using alpha composite
        blended = Image.alpha_composite(image, red_mask)
        
        # Add space for caption
        caption_height = 50
        new_image = Image.new("RGBA", (blended.width, blended.height + caption_height), (255, 255, 255, 255))
        new_image.paste(blended, (0, caption_height))
        
        # Draw caption
        draw = ImageDraw.Draw(new_image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position for centering
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = (new_image.width - text_width) // 2
        text_y = (caption_height - text_height) // 2
        
        # Draw text
        draw.text((text_x, text_y), caption, fill="black", font=font)
        
        return new_image
