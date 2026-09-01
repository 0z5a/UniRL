import os
from pathlib import Path
from PIL import Image
from loguru import logger
import torchvision.transforms as transforms
import torch
from hymm.data_kits.csv_dataset import CSVDataset
from index_kits import ArrowIndexV2
from easydict import EasyDict
try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None
from hymm.data_kits.instruction_tuning_transfusion_loader import InstructionTuningTransfusionArrowStream
from hymm.utils.file_utils import log_in_safe_logger

class CSVImageFolderDataset(CSVDataset):
    def __init__(
        self,
        source_dir,
        save_template,
        image_size=None,  # will resize src image read from bytes
        csv_name="prompts.csv",
        src_folder_name="src_image",
        tgt_folder_name="tgt_image",
        image_suffix=".png",
        face_crop=False,
        src_condition_type=["vae"],
        **kwargs
    ):
        # Convert to Path object
        source_dir = Path(source_dir)
        
        # Construct paths
        self.image_size = image_size
        if kwargs.get("logger", None) is None:
            kwargs["logger"] = logger
        self.logger = kwargs["logger"]
        csv_path = source_dir / csv_name
        self.src_image_dir = source_dir / src_folder_name
        self.tgt_image_dir = source_dir / tgt_folder_name
        self.image_suffix = image_suffix
        # Setup default args if not provided
        self.face_crop = face_crop
        self.src_condition_type = src_condition_type
        
        self.src_image_list = list(self.src_image_dir.glob(f"*{self.image_suffix}"))
        self.tgt_image_list = list(self.tgt_image_dir.glob(f"*{self.image_suffix}"))
        self.logger.info(f"len(self.src_image_list): {len(self.src_image_list)}")
        self.logger.info(f"len(self.tgt_image_list): {len(self.tgt_image_list)}")

        # Validate that directories and csv exist
        self._validate_paths(csv_path)
        
        # Initialize face analysis if needed
        if self.face_crop is True:
            self._initialize_face_analysis()

        # Initialize parent class with csv path
        super().__init__(source=str(csv_path), save_template=save_template, **kwargs)

    def _validate_paths(self, csv_path):
        """Validate that directories and CSV exist."""
        if not csv_path.exists():
            raise ValueError(f"CSV file does not exist: {csv_path}")
        if not self.src_image_dir.exists():
            raise ValueError(f"Source image directory does not exist: {self.src_image_dir}")
        if not self.tgt_image_dir.exists():
            self.logger.warning(f"Target image directory does not exist: {self.tgt_image_dir}")

    def _initialize_face_analysis(self):
        """Initialize face analysis model."""
        ASSETS_BASE = os.getenv("ASSETS_BASE", "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets").rstrip('/')
        insightface_path = f"{ASSETS_BASE}/others/insightface"
        name = 'buffalo_l'  # From large to small, support antelopev2, buffalo_l, buffalo_sc
        allowed_modules = ['detection']
        if "face_embed" in self.src_condition_type:
            allowed_modules.append('recognition')
        self.logger.info(f"Loading face analysis, name: {name}, allowed_modules: {allowed_modules}, providers CPUExecutionProvider, from {insightface_path}")
        self.face_analysis = FaceAnalysis(name=name, root=insightface_path, allowed_modules=allowed_modules, providers=['CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        
    def _process_face_crop(self, image, image_path, is_source, prefix):
        """Process an image with face cropping."""
        image_tensor = transforms.ToTensor()(image)
        log_in_safe_logger(image_tensor, self.logger, f'{prefix}_image_tensor of raw image before crop at path: {image_path}')
        
        # Convert to [-1, 1] range
        image_tensor_norm = image_tensor * 2.0 - 1.0
        
        # Crop and get face embedding
        image_tensor_norm, face_embedding, face_bbox = InstructionTuningTransfusionArrowStream.crop_face_image_torch(
            self.face_analysis, 
            image_tensor_norm, 
            max_side=self.image_size, 
            logger=self.logger, 
            src_condition_type=self.src_condition_type,
            return_bbox=True
        )
        
        # Convert back to [0, 1] range
        image_tensor = (image_tensor_norm + 1.0) / 2.0
        
        log_in_safe_logger(image_tensor, self.logger, f'{prefix}_image_tensor after face crop')
        log_in_safe_logger(face_embedding, self.logger, f'{prefix}_face_embedding after face crop')
        
        return image_tensor, face_embedding, face_bbox

    def _process_image(self, data_dict, image_path, is_source=True):
        """Process an image (either source or target) and add it to the data dict."""
        prefix = "src" if is_source else "tgt"
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            if self.face_crop is True:
                # Process with face crop
                image_tensor, face_embedding, face_bbox = self._process_face_crop(
                    image, image_path, is_source, prefix
                )
                
                # Handle face embeddings and bounding boxes
                if face_embedding is not None:
                    data_dict[f"{prefix}_face_embedding"] = face_embedding
                    # Validate source face embedding
                    if is_source and torch.all(face_embedding == 0) and "face_embed" in self.src_condition_type:
                        raise ValueError(f"{prefix}_face_embedding is torch.zeros(512), at path: {image_path}, not a valid face image; please remove it")
                    # Log warning for target face embedding
                    elif not is_source and torch.all(face_embedding == 0):
                        print(f"{prefix}_face_embedding is torch.zeros(512), at path: {image_path}, not a valid face image, skip this image; please remove it")
                
                if face_bbox is not None:
                    data_dict[f"{prefix}_face_bbox"] = face_bbox
                    
            elif self.image_size is not None:
                # Process with resize and crop
                resized_image, crop_info = ArrowIndexV2.resize_and_crop(image, target_size=self.image_size, crop_type='center')
                image_tensor = transforms.ToTensor()(resized_image)
                data_dict["crop_info"] = crop_info
            else:
                # Just convert to tensor
                image_tensor = transforms.ToTensor()(image)
            
            data_dict[f"{prefix}_image"] = image_tensor
            data_dict[f"{prefix}_image_path"] = str(image_path)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return data_dict

    def __getitem__(self, index):
        # Get the base prompt dict from parent class
        data_dict = super().__getitem__(index)
        data_dict["prompt"] = data_dict["input"]
        
        # Generate image paths
        id_str = f"{(data_dict['id'] % len(self.src_image_list)):04d}"
        src_image_path = self.src_image_dir / f"{id_str}{self.image_suffix}"
        tgt_image_path = self.tgt_image_dir / f"{id_str}{self.image_suffix}"

        # Process source image
        data_dict = self._process_image(
            data_dict, 
            src_image_path, 
            is_source=True
        )
        
        # Process target image if it exists
        if tgt_image_path.exists():
            data_dict = self._process_image(
                data_dict, 
                tgt_image_path, 
                is_source=False
            )

        return data_dict

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data samples into a single batch dictionary.
        
        Args:
            batch: List of data dictionaries from __getitem__
            
        Returns:
            dict: Collated batch with stacked tensors
        """
        # Start with basic metadata collection
        collated = {
            "ids": [item["id"] for item in batch],
            "seeds": [item["seed"] for item in batch],
            "prompt": [item["prompt"] for item in batch],
        }

        # Process required image data
        collated.update({
            "src_image": torch.stack([item["src_image"] for item in batch]),
            "src_image_path": [item["src_image_path"] for item in batch],
        })
        
        # Process optional tensor fields
        CSVImageFolderDataset._maybe_add_tensor_field(batch, collated, "tgt_image")
        CSVImageFolderDataset._maybe_add_field(batch, collated, "tgt_image_path")
        
        # Process face embeddings
        CSVImageFolderDataset._maybe_add_tensor_field(batch, collated, "src_face_embedding")
        CSVImageFolderDataset._maybe_add_tensor_field(batch, collated, "tgt_face_embedding")
        
        # Process face bounding boxes
        CSVImageFolderDataset._maybe_add_tensor_field(batch, collated, "src_face_bbox")
        CSVImageFolderDataset._maybe_add_tensor_field(batch, collated, "tgt_face_bbox")
        
        # Process crop info
        CSVImageFolderDataset._maybe_add_tensor_field(batch, collated, "crop_info")
        
        return collated

    @staticmethod
    def _maybe_add_tensor_field(batch, collated, field_name):
        """
        Add a tensor field to the collated batch if it exists in the first item,
        stacking the tensors from all items. Ensures values are converted to tensors.
        
        Args:
            batch: List of data dictionaries
            collated: Dictionary to update with the collated field
            field_name: Name of the field to collate
        """
        if field_name in batch[0]:
            tensors = [torch.tensor(item[field_name]) if not isinstance(item[field_name], torch.Tensor)
                      else item[field_name] for item in batch]
            collated[field_name] = torch.stack(tensors)
    
    @staticmethod
    def _maybe_add_field(batch, collated, field_name):
        """
        Add a non-tensor field to the collated batch if it exists in the first item.
        
        Args:
            batch: List of data dictionaries
            collated: Dictionary to update with the collated field
            field_name: Name of the field to collate
        """
        if field_name in batch[0]:
            collated[field_name] = [item[field_name] for item in batch]


if __name__ == "__main__":
    
    
    # create a default args dict
    args = {
        "face_crop": True,
        "src_condition_type": "face_embed",
    }
    # use easydict to create a default args dict
    from easydict import EasyDict
    args = EasyDict(args)
    
    
    save_base = Path("/apdcephfs_cq8/share_2938211/chenyangqi/hymm_text2image_ar/__trash/data/")
    # id_csv_image_folder = '/apdcephfs_cq8/share_2938211/chenyangqi/datasets_gy2/image_id_preserve/testsets/src_tgt_prompt_512'
    id_csv_image_folder = '/apdcephfs_cq8/share_2938211/chenyangqi/datasets_gy2/image_id_preserve/testsets/pm_iid_bench'
    save_template = str(save_base / ("{}_{{}}" + ".png"))
    from loguru import logger
    dataset = CSVImageFolderDataset(
        source_dir = id_csv_image_folder,
        save_template=save_template,
        image_size=(256, 256),
        skip_exist=False,
        logger=logger,
        face_crop=args.face_crop,
        src_condition_type=args.src_condition_type,
    )
    logger.info(f"len(dataset): {len(dataset)}")
    item_0 = dataset[0]
    log_in_safe_logger(item_0, logger, "item_0")
        
    # get src_face_embedding and tgt_face_embedding
    src_face_embedding = item_0['src_face_embedding']
    tgt_face_embedding = item_0['tgt_face_embedding']

    
    # cosine similarity between src_face_embedding and tgt_face_embedding
    cosine_similarity = torch.nn.functional.cosine_similarity(src_face_embedding, tgt_face_embedding, dim=0)
    logger.info(f"cosine similarity between src_face_embedding and tgt_face_embedding: {cosine_similarity}")
    
    
    # calculate the similarity between src_image and tgt_image in whole dataset
    cosine_similarity_list = []
    for item in dataset:
        src_image = item['src_face_embedding']
        tgt_image = item['tgt_face_embedding']
        cosine_similarity = torch.nn.functional.cosine_similarity(src_image, tgt_image, dim=0)
        logger.info(f"cosine similarity between src_image and tgt_image: {cosine_similarity}")
        if torch.all(cosine_similarity == 0):
            logger.info(f"cosine similarity between src_image and tgt_image is 0, skip this item")
        else:
            cosine_similarity_list.append(cosine_similarity)

    logger.info(f"average cosine similarity between src_image and tgt_image: {sum(cosine_similarity_list) / len(cosine_similarity_list)}")
    
    

# DEBUG=true PYTHONPATH=./ python3 hymm/data_kits/csv_folder_dataset.py