from functools import partial
from typing import List, Optional
from argparse import Namespace
import os
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from transformers import PreTrainedModel, PreTrainedTokenizer

from .configuration_emu import EmuConfig
from .constants import *
from .modeling_llama import LlamaForCausalLM
from .visual import EVAVisionTransformer
from hymm.utils.file_utils import log_in_safe_logger

class EmuPreTrainedModel(PreTrainedModel):
    config_class = EmuConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer", "Block"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# leave the model here so we can use it in the next implementation of MAR-style model
class EmuForClsAndRegression(EmuPreTrainedModel):

    def __init__(self, config):
        super(EmuForClsAndRegression, self).__init__(config)
        print("Loading LlamaForCausalLM model ")
        self.lm = LlamaForCausalLM(config=config)

        self.lm.model.embed_tokens.padding_idx = config.pad_token_id

    def get_num_layers(self):
        return len(self.lm.model.layers)

from torchvision import transforms as TF

EVA_IMAGE_SIZE = 448
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

class EmuModel(EmuPreTrainedModel):

    def __init__(self, config, logger=None):
        super().__init__(config)

        self.eva_size=EVA_IMAGE_SIZE
        self.eva_mean=OPENAI_DATASET_MEAN
        self.eva_std=OPENAI_DATASET_STD
        self.logger = logger
        self._saved_logger = logger

        vision_config = Namespace(**config.vision_config)
        self.pooling_stride = config.pooling_stride
        self.visual = EVAVisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.width // vision_config.head_width,
            mlp_ratio=vision_config.mlp_ratio,
            qkv_bias=vision_config.qkv_bias,
            drop_path_rate=vision_config.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=vision_config.layer_norm_eps),
            # xattn=vision_config.xattn,
            xattn=False, # FIXME xformer need to be replace with scaled dot product attention from pytorch
            postnorm=vision_config.postnorm,
        )
        # print("Skip EmuForClsAndRegression model in Emu2-Gen.multimoda_encoder")
        # self.decoder = EmuForClsAndRegression(config)

        self.gradient_checkpointing = False
        
        self.n_query = vision_config.n_query
        self.v_query = vision_config.v_query

        self.transform01_to_normal = TF.Compose([
            TF.Resize((self.eva_size, self.eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            # TF.ToTensor(),
            TF.Normalize(mean=self.eva_mean, std=self.eva_std),
        ])
        # tensor.sub_(mean).div_(std)
        self.transform_11_to_normal = TF.Compose([
            TF.Normalize(mean=-1.0, std=2.0),
            TF.Resize((self.eva_size, self.eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.Normalize(mean=self.eva_mean, std=self.eva_std),
        ])
    def set_logging_enabled(self, enabled=True):
        """Temporarily enable/disable logging"""
        self.logger = self._saved_logger if enabled else None
    @property
    def device(self):
        return next(iter(self.parameters())).device


    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def transform(self, image, input_range):
        if input_range == "01":
            return self.transform01_to_normal(image)
        elif input_range == "-11":
            return self.transform_11_to_normal(image)
        elif input_range == "normal":
            return image
        else:
            raise ValueError(f"Input type {input_range} is not supported")
    @torch.no_grad()
    def check_input(self, image, logger=None):
        n, c, h, w =image.shape
        assert (h==448 and w==448), f"Input image size should be 448x448, but got {h}x{w}"
        warning_str = ''
        mean = image.mean()
        std = image.std()
        if mean < -1.0 or mean > 1.0:
            warning_str += f"EVACLIP mean should be 0, but {mean:.2f}; "
        if std > 1.5 or std < 0.1:
            warning_str += f"EVACLIP std should be 1, but {image.std().item():.2f};"
        if warning_str != '':
            print(warning_str)
            # check if logger has handlers; logger can be None, empty logger, or logger with multiple handlers
            if logger is not None and hasattr(logger, "_core") and len(logger._core.handlers) > 1:
                logger.warning(warning_str)


    @torch.no_grad()
    def encode_image(self, image: torch.Tensor, *, n_query=None, pooling_stride=None, logger=None):
        """
        Encode the image to the image embedding.
        Args:
            image (torch.Tensor): The input image tensor. shape: [1, 3, H, W], min: -2.05, max: 2.44, mean: 0.02, std: 0.98
            n_query (int, optional): The number of query tokens. Defaults to None.
            pooling_stride (int, optional): The pooling stride. Defaults to None.
            logger (logging.Logger, optional): The logger to print warning of input range. Defaults to None.
        """

        # Check image 
        if logger is None: logger = self.logger
        if logger is not None: self.check_input(image, logger=logger)
        n_query = n_query if n_query is not None else self.n_query

        image_embeds = self.visual(image)
        
        image_embeds = image_embeds[:, 1:, :]
        b, n, c = image_embeds.shape
        sqrt_n = int(n**0.5)
        image_embeds = image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
        if pooling_stride is None:
            pooling_stride = self.pooling_stride
        if pooling_stride is None:
            # 448 pixel -> 32 patch -> pooling to 8
            stride = int(sqrt_n // (n_query ** 0.5))
            if logger is not None: logger.info(f"Calculate Pooling stride {stride} from sqrt_n: {sqrt_n}, n_query: {n_query}") 
            
        else:
            stride = pooling_stride
            if logger is not None: logger.info(f"Set Pooling stride from args: {stride}, get number of tokens: {n//stride//stride}")

        image_embeds = F.avg_pool2d(image_embeds, kernel_size=(stride, stride), stride=stride)
        image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
        # keep 2 decimal places for logging
        
        log_in_safe_logger(image_embeds, logger, "image_embeds after pooling in encode_image")
        return image_embeds


class EmuForCausalLM(EmuPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config, logger=None):
        super().__init__(config)

        self.config = config
        self.model = EmuModel(config, logger=logger)
        
        self.logger = logger
        self._saved_logger = logger
        # LM to EVA
        self.project_down = nn.Linear(config.hidden_size, config.d_model, bias=False)
        # EVA to LM
        self.project_up = nn.Linear(config.d_model, config.hidden_size, bias=False)

        self.n_query = self.model.n_query
        self.image_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * self.n_query + DEFAULT_IMG_END_TOKEN
    
    def set_logging_enabled(self, enabled=True):
        """Temporarily enable/disable logging"""
        self.logger = self._saved_logger if enabled else None
        self.model.set_logging_enabled(enabled=enabled)

    def device(self, module=None):
        if module is None:
            return next(self.parameters()).device
        return next(module.parameters()).device

    def dtype(self, module):
        if module is None:
            return next(self.parameters()).dtype
        return next(module.parameters()).dtype

    # not used in our first version
    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: PreTrainedTokenizer,
        image: Optional[torch.Tensor] = None,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        IMAGE, BOI = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        if image is not None:
            prompt_image_embeds = self.model.encode_image(image)
            _, _, c = prompt_image_embeds.shape
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            prompt_image_embeds = self.project_up(prompt_image_embeds)

        text = [t.replace(placeholder, self.image_placeholder) for t in text]

        target_image_embeds = None
        for num_img_token in range(self.n_query):
            if num_img_token == 0:
                text = [f"{t}{DEFAULT_IMG_TOKEN}" for t in text]
            else:
                text = [f"{t}{DEFAULT_IMAGE_TOKEN}" for t in text]

            inputs = tokenizer(text, padding="longest", return_tensors="pt")
            device = self.device(self.model.decoder.lm.model.embed_tokens)
            attention_mask = inputs.attention_mask.to(device)
            input_ids = inputs.input_ids.to(device) # B x N

            text_embeds = self.model.decoder.lm.model.embed_tokens(input_ids)

            image_idx = (input_ids == IMAGE)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            if image is not None:
                prompt_idx = torch.logical_and(image_idx, cumsum_idx > num_img_token)
                text_embeds[prompt_idx] = prompt_image_embeds.to(text_embeds.device)

            if target_image_embeds is not None:
                target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token))
                text_embeds[target_idx] = self.project_up(target_image_embeds).to(text_embeds.device)

            outputs = self.model.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            image_idx = (input_ids == IMAGE) + (input_ids == BOI)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token+1))

            hidden_states = outputs.hidden_states[-1]
            target_image_embeds = hidden_states[target_idx.to(hidden_states.device)]
            target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
            target_image_embeds = self.project_down(target_image_embeds)

        _, C = target_image_embeds.shape
        B = hidden_states.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)

        return target_image_embeds

