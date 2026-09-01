import os
import re
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse


aes_decrypt: Optional[Callable] = None
FILE_PASSWORD: Optional[str] = None
HEX_CHAR: str = '0123456789abcdef'


class DataMixin:
    """A mixin class for data handling utilities."""

    enable_crypto: bool = False
    cos_base: Optional[list[str]] = None

    def setup_data(self, enable_crypto=False, cos_base=None, verbose=1):
        self.enable_crypto = enable_crypto
        self.cos_base = cos_base

        # Check for crypto utilities and load FILE_PASSWORD
        if enable_crypto:
            global aes_decrypt, FILE_PASSWORD
            try:
                from hymm.utils.crypto_utils import aes_decrypt
            except (ModuleNotFoundError, ImportError) as e:
                raise ModuleNotFoundError(
                    f"`enable_crypto` is enabled, but crypto module not found. {str(e)}"
                )

            # Try to load password from environment variable
            FILE_PASSWORD = os.getenv('FILE_PASSWORD')
            if FILE_PASSWORD is None or FILE_PASSWORD == "":
                raise ValueError(f"`enable_crypto` is enabled, but `FILE_PASSWORD` environment variable is not set.")
            elif verbose:
                print(f"FILE_PASSWORD loaded: {FILE_PASSWORD[:4]}...")

        # Initialize cos_base sources and targets
        self.cos_base_sources, self.cos_base_targets, self._cos_required_keys, self.cos_base_source_patterns = self.parse_cos_base(cos_base)

    # ===========================
    #      Helper Functions
    # ===========================
    @staticmethod
    def require_configs(obj, required, obj_name, do_assert=True):
        if isinstance(required, str):
            required = [required]

        # Use tuple for alternatives
        if isinstance(required, tuple):
            passed = DataMixin.require_configs(obj, required[0], None, do_assert=False)
            if not passed:
                for alt_required in required[1:]:
                    passed = DataMixin.require_configs(obj, alt_required, None, do_assert=False)
                    if passed:
                        break
                else:
                    raise KeyError(f"One of {required} is required for {obj_name}.")
            return passed

        else:
            missing_keys = []
            if isinstance(obj, (dict, list, tuple, set)):
                for key in required:
                    if key not in obj:
                        missing_keys.append(key)
            else:
                for key in required:
                    if not hasattr(obj, key) or getattr(obj, key) is None:
                        missing_keys.append(key)
            if do_assert and len(missing_keys) > 0:
                raise KeyError(f"[{', '.join(missing_keys)}] is required for {obj_name}.")
            return len(missing_keys) == 0

    # ===========================
    #       COS Utilities
    # ===========================
    @staticmethod
    def parse_cos_url(url_cos, content):
        parsed_url = urlparse(url_cos)
        if content == "bucket":
            ret = parsed_url.netloc.split('.')[0]
        elif content == "key":
            ret = parsed_url.path.lstrip('/')
        elif content == "bucket,key":
            ret = (parsed_url.netloc.split('.')[0], parsed_url.path.lstrip('/'))
        else:
            raise ValueError(f"Invalid content: {content}")
        return ret

    @staticmethod
    def parse_cos_base(cos_base):
        required_keys = set()
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
                    # Escape the source for regex, then replace {stem2} with a pattern
                    # that matches any single directory component
                    escaped = re.escape(src).replace(re.escape('{stem2}'), '[^/]+')
                    cos_base_source_patterns.append(re.compile(escaped))
                else:
                    cos_base_source_patterns.append(None)
                if '{bucket}' in cos_base_target:
                    required_keys.add("url_cos")
            return cos_base_sources, cos_base_targets, required_keys, cos_base_source_patterns
        else:
            return None, None, required_keys, None

    def parse_cos_path(self, image_path, **kwargs):
        """ It handles several cases:
        1. `image_path` is a cos path and contains special characters like '%20',
            which need to be unquoted.
        2. Replace the cos_base if necessary
        3. Append '.enc' if the cos files are encrypted
        4. Fill the bucket if needed
        5. Remove the sign if the cos path contains '?sign=' and is not a http url
        """
        image_path = unquote(image_path)

        if self.cos_base_sources is not None:
            for src_base, tgt_base, src_pattern in zip(
                self.cos_base_sources, self.cos_base_targets, self.cos_base_source_patterns
            ):
                if src_pattern is not None:
                    # Source contains {stem2} wildcard, use regex to match and remove
                    # the 2-char hex directory from the path
                    m = src_pattern.match(image_path)
                    if m:
                        # Replace the matched source prefix with the target
                        image_path = tgt_base + image_path[m.end():]
                        break
                elif image_path.startswith(src_base):
                    image_path = image_path.replace(src_base, tgt_base)
                    break

        # ===================================
        # Fill path placeholders
        values = dict()

        # Fill bucket if needed
        if "{bucket}" in image_path:
            assert "url_cos" in kwargs, f"`url_cos` is required to parse cos path."
            bucket = self.parse_cos_url(kwargs["url_cos"], "bucket")
            values["bucket"] = bucket

        # Insert the first two characters of the stem in the path prefix
        if "{stem2}" in image_path:
            stem2 = Path(image_path).stem[:2]
            if stem2[0] in HEX_CHAR and stem2[1] in HEX_CHAR:
                values["stem2"] = stem2
            else:
                values["stem2"] = "GG"

        if len(values) > 0:
            image_path = image_path.format(**values)

        # =================================================
        # 处理cos图片的签名
        if "http" not in image_path and "?sign=" in image_path:
            # self.logger.warning(f"Cos image path: {image_path} has sign, which will be removed.")
            image_path = image_path.split("?sign=")[0]

        # Append '.enc' if not exists
        if self.enable_crypto:
            if not image_path.endswith('.enc') and not Path(image_path).exists():
            # if not image_path.endswith('.enc'):
                image_path = image_path + '.enc'
        else:
            if image_path.endswith('.enc'):
                image_path = image_path[:-4]

        return image_path

    # ==================================
    #   File Encryption and Decryption
    # ==================================
    @staticmethod
    def decrypt_file(file_path):
        assert FILE_PASSWORD is not None, "`FILE_PASSWORD` must be provided when reading an encrypted file"
        file_buffer = aes_decrypt(file_path, FILE_PASSWORD)
        return file_buffer
