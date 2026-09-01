import re
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, ImageOps

from .index_dataset import IndexDataset
from ..utils.helpers import to_2tuple


class ImageDataset(IndexDataset):
    def __init__(
            self,
            index_file,
            multireso=False,
            batch_size=1,
            world_size=1,
            base_size=None,
            logger=None,
            debug=False,
            args=None,
    ):
        super().__init__(
            index_file=index_file,
            multireso=multireso,
            batch_size=batch_size,
            world_size=world_size,
            logger=logger,
            debug=debug,
        )
        self.args = args

        # image meta info
        self.base_size = to_2tuple(base_size)
        self.training_image_size = self.base_size   # bc
        self.base_height, self.base_width = self.base_size

        # image getter
        self.cos_base = self.args.get('cos_base', None)
        self.cos_base_sources, self.cos_base_targets, self.cos_base_source_patterns = self.parse_cos_base(self.cos_base)
        self.cos_file_is_encrypted = self.args.get('cos_file_is_encrypted', False)
        self.enable_resample = self.args.get('enable_resample', False)

    @staticmethod
    def parse_cos_base(cos_base):
        if cos_base is not None:
            if isinstance(cos_base, str):
                cos_base = [cos_base]
            cos_base_sources = []
            cos_base_targets = []
            cos_base_source_patterns = []
            for cos_base_i in cos_base:
                assert "->" in cos_base_i, "cos_base should be in the format of 'old_base -> new_base'"
                cos_base_source, cos_base_target = cos_base_i.split("->")
                cos_base_sources.append(cos_base_source.strip())
                cos_base_targets.append(cos_base_target.strip())
                # Build regex pattern for source paths containing {stem2} wildcard
                src = cos_base_source.strip()
                if '{stem2}' in src:
                    escaped = re.escape(src).replace(re.escape('{stem2}'), '[^/]+')
                    cos_base_source_patterns.append(re.compile(escaped))
                else:
                    cos_base_source_patterns.append(None)
            return cos_base_sources, cos_base_targets, cos_base_source_patterns
        else:
            return None, None, None

    def parse_image_path(self, src, image_path):
        '''
            It handles several cases:
            1. `image_path` is a cos path and contains special characters like '%20',
            which need to be unquoted.
            2. Replace the cos_base if necessary
            3. Append '.enc' if the cos files are encrypted
            4. Fill the bucket if needed
            5. Remove the sign if the cos path contains '?sign=' and is not a http url
        '''

        image_path = unquote(image_path)

        if self.cos_base is not None:
            for src_base, tgt_base, src_pattern in zip(
                self.cos_base_sources, self.cos_base_targets, self.cos_base_source_patterns
            ):
                if src_pattern is not None:
                    m = src_pattern.match(image_path)
                    if m:
                        image_path = tgt_base + image_path[m.end():]
                        break
                elif image_path.startswith(src_base):
                    image_path = image_path.replace(src_base, tgt_base)
                    break
        if self.cos_file_is_encrypted:
            image_path = image_path + '.enc'
        
        # Fill bucket if needed
        if "{bucket}" in image_path:
            if isinstance(src, (int, np.integer)):
                url_cos = self.index_manager.get_attribute(src, **self.extra_url_cos_col)
            elif isinstance(src, dict):
                url_cos = src[self.extra_url_cos_col.key]
            else:
                raise ValueError(f"Unsupported src type: {type(src)}")
            bucket = self.get_from_url_cos(url_cos, "bucket")
            image_path = image_path.format(bucket=bucket)
        
        # 处理cos图片的签名
        if "http" not in image_path and "?sign=" in image_path:
            # self.logger.warning(f"Cos image path: {image_path} has sign, which will be removed.")
            image_path = image_path.split("?sign=")[0]

        return image_path

    def get_raw_image(self, src, real_index=None, convert_mode="RGB", apply_exif=False, **image_col):
        image_path = None
        try:
            if 'cache_image' in image_col['column']:
                if isinstance(src, (int, np.integer)):
                    image_path = self.index_manager.get_attribute(src, **image_col)
                elif isinstance(src, str):
                    image_path = src
                elif isinstance(src, dict):
                    image_path = src[image_col['column']]
                else:
                    raise ValueError(f"Unsupported src type: {type(src)}")
                # TODO hard code
                if not '/apdcephfs_wza/share_303937731/yijicheng/tmp_training_data/cache_image/' in image_path and not '/apdcephfs_wza/branzkjiang/sft_data/imgs' in image_path and not '/apdcephfs_wza/branzkjiang/reward_models_data/hunyuan_edit' in image_path:
                    image_path = self.parse_image_path(src, image_path)
                image = self.read_local_image(image_path, convert_mode=convert_mode, apply_exif=apply_exif)
            elif image_col['column'] == "url_bucket_key":
                image_path = src[self.extra_url_cos_col.key]
                parsed_url = urlparse(image_path)
                key = parsed_url.path.lstrip('/')
                image_path = f"/cos_nj1/share_302243067/{key}"
                image_path = self.parse_image_path(src, image_path)
                image = self.read_local_image(image_path, convert_mode=convert_mode, apply_exif=apply_exif)
            elif "url" in image_col['column']:
                image = self.get_image_from_url_cos(src, apply_exif=apply_exif, **image_col)
            else:
                image = self.get_image_from_arrow(src, apply_exif=apply_exif, **image_col)
            image_flag = "normal"
        except Exception as e:
            index = src if isinstance(src, (int, np.integer)) else real_index
            self.logger.error(f"({image_col=}, {index=}, {image_path=}) {type(e)}: {e}. Fallback to gray image.")
            image = Image.new("RGB", (self.base_size[0], self.base_size[1]), (128, 128, 128))
            image_flag = "gray"
        return image, image_flag
