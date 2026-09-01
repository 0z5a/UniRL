from collections import defaultdict

import torch
import torchvision.transforms as transforms
from PIL import Image
# index_kits is a self-developed library for easily loading and processing large-scale datasets
# More details: https://iwiki.woa.com/p/4010805183
# Source code and doc: https://git.woa.com/CreativeAd/IndexKits

from hymm.utils.helpers import to_2tuple, default
from hymm.models import TokenizerWrapper
from .index_dataset import IndexDataset
from ..constants import VAE_META_INFO
from hymm.samplers.image_processor_vlm import VLMImageProcessor


class JanusTextImageArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=256,
        image_token_length=1024,
        image_token_offset=0,
        use_pre_extracted_token=False,
        text_token_length=256,
        uncond_p=0.0,
        patch_size=1,
        tokenizer_name=None,
        raise_text_error=False,
        multireso=False,
        add_iw_ih_token=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
    ):
        super().__init__(
            index_file,
            multireso,
            index_kwargs.get("batch_size", 1),
            index_kwargs.get("world_size", 1),
            logger,
            debug,
        )
        self.args = args
        self.training_image_size = to_2tuple(training_image_size)
        self.uncond_p = uncond_p
        self.add_iw_ih_token = add_iw_ih_token
        self.raise_text_error = raise_text_error

        self.use_pre_extracted_token = use_pre_extracted_token

        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = patch_size

        self.enable_think_mode = self.args.get("enable_think_mode", False)
        if self.enable_think_mode:
            self.system_prompt = (
                "The user provides a prompt, and the assistant generates images and verifies whether the number of objects "
                "in the generated images is accurate. The assistant first thinks about the reasoning process in the mind "
                "and then provides the user with the answer. The reasoning process and the final generated image are enclosed "
                "within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
                "<answer> the generated image </answer>\n\nUser: Generate an image of '{}'. Assistant: "
            )

        if index_kwargs is None:
            index_kwargs = {}

        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=index_kwargs.get("ceph_base", None),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )

        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    Using {self.index_manager}")
      
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

        # for understanding task
        self.und_image_preprocessor = VLMImageProcessor.from_pretrained(args.image_preprocessor_pretrained_path)

        self.logger.info(f"Image transform: {self.pil_image_to_tensor}")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100
        # tokenizer
        self.text_token_length = text_token_length
        self.image_token_length = image_token_length
        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            patch_special_tokens_dict = {
            "img": args.get("img_token_tag", None),
            "pad": args.get("pad_token_tag", None),
            "cfg": args.get("cfg_token_tag", None),
            "boi": args.get("boi_token_tag", None),
            "eoi": args.get("eoi_token_tag", None),
        }
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger, patch_special_tokens_dict)
        else:
            self.tokenizer = tokenizer

        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        pass

    def get_raw_images(self, index):
        image_paths = self.index_manager.get_attribute(index, column="paths")
        ret = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            ret.append(img)
        return ret

    def get_images_with_size(self, index):
        images = self.get_raw_images(index)

        origin_size = images[0].size  # (w_ori, h_ori)
        target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        images_tensor = []
        for image in images:
            image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
                image, target_size, crop_type="random", resample=Image.Resampling.BICUBIC
            )
            image_tensor = self.pil_image_to_tensor(image)
            images_tensor.append(image_tensor)
        if self.enable_think_mode:
            # <answer> </answer> 中出最终图，所以需要重复最后一张图
            images_tensor.append(image_tensor)
        images_tensor = torch.stack(images_tensor, dim=0)  # [N, C, H, W]
        # for image understanding
        und_images_tensor = self.und_image_preprocessor(images, return_tensors="pt").pixel_values # [N, C, H, W]
        # padding for collate a batch
        # TODO: 目前固定为最大长度3
        eff_images_num = und_images_tensor.shape[0]
        if und_images_tensor.shape[0] < 3:
            pad_tensor = torch.zeros(3 - und_images_tensor.shape[0], *und_images_tensor.shape[1:])
            und_images_tensor = torch.cat([und_images_tensor, pad_tensor], dim=0)
            max_gen_images_num = 4 if self.enable_think_mode else 3
            pad_tensor = torch.zeros(max_gen_images_num - images_tensor.shape[0], *images_tensor.shape[1:])
            images_tensor = torch.cat([images_tensor, pad_tensor], dim=0)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return images_tensor, und_images_tensor, eff_images_num, kwargs

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_text(self, ind):
        cot_caption = self.index_manager.get_attribute(ind, column="cot_caption")
        text = str(cot_caption).strip()

        if self.enable_think_mode:
            img_prompt = text.split("User: Generate image of '")[1].split("'\n\nAssistant")[0]
            prompt = self.system_prompt.format(img_prompt)
            answer = text.split("'\n\nAssistant:")[1]
            text = prompt + answer

        return text

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

        # Get text
        text = self.get_text(index)

        # Get image
        if self.use_pre_extracted_token:
            raise NotImplementedError
        else:
            images, und_images_tensor, eff_images_num, kwargs = self.get_images_with_size(index)
            image_height, image_width = images.shape[-2:]
            tk_height = image_height // self.downsample_factor[0] // self.patch_size
            tk_width = image_width // self.downsample_factor[1] // self.patch_size
            cur_image_token_length = int(tk_height * tk_width)
            image_token_shape_wh = torch.tensor([tk_width, tk_height], dtype=torch.long)
            assert cur_image_token_length == self.image_token_length

            tokens, text_loss_mask, image_loss_mask = self.tokenizer.encode_ar_janus_cot(
                text,
                max_n_images=3,
                max_text_token_length=self.text_token_length + 1,
                max_image_token_length=self.image_token_length,   # max_token_length_per_image
                uncond_p=self.uncond_p,
                uncond_enabled=True,
                image_token_length=cur_image_token_length,
                enable_think_mode=self.enable_think_mode,
            )
            target_token = tokens.clone() # will not be used

        # Prepare kwargs
        kwargs["index"] = index

        ret = {
            "eff_images_num": eff_images_num,
            "images": images,
            "und_images_tensor": und_images_tensor,
            "tokens": tokens,
            "target_token": target_token,
            "text_loss_mask": text_loss_mask,
            "image_loss_mask": image_loss_mask,
            "text": text,
            "object": self.index_manager.get_attribute(index, column="object"),
            "kwargs": {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        }

        return ret
