import random
import json
import math
from typing import Dict, Any, List
from loguru import logger

from .caption_base import CaptionOut
from ...utils.resolution import ratio_index_repr
from ...utils.resolution_2 import resolution_repr_fn

PROMPT_PATCHERS = {}
COMMA = {"zh": lambda: random.choice(["，", "。"]), "en": lambda: random.choice([", ", ",", ". ", "."])}
PERIOD = {"zh": "。", "en": "."}


def register_patcher(func):
    PROMPT_PATCHERS[func.__name__] = func
    return func


def load_resource(patcher_names: List, **kwargs):
    resource = {}
    if "image_ratio" in patcher_names:
        if "image" in kwargs:
            assert hasattr(kwargs["image"], "i"), \
                f"image tensor must have attribute 'i' for image_ratio patcher"
            resource["ratio_index"] = kwargs["image"].i.ratio_index
        elif "ratio_index" in kwargs:
            resource["ratio_index"] = kwargs["ratio_index"]
        else:
            raise ValueError("image_ratio patcher requires 'image'(ImageTensor) or 'ratio_index'(int) in kwargs")
    
    if "image_resolution" in patcher_names:
        if "image" in kwargs:
            assert hasattr(kwargs["image"], "i"), \
                f"image tensor must have attribute 'i' for image_ratio patcher"
            resource["image_width"] = kwargs["image"].i.image_width
            resource["image_height"] = kwargs["image"].i.image_height
        else:
            raise ValueError("image_resolution patcher requires 'image'(ImageTensor) in kwargs")

    if "fps" in patcher_names:
        dataset = kwargs["dataset"]
        index = kwargs["index"]
        dataset.require_configs(dataset.index_columns, "fps_col",
                                f"{dataset.dataset_tag}.{dataset.dataset_tag}_index_kwargs.index_columns")
        fps = dataset.index_manager.get_attribute(index, **dataset.index_columns["fps_col"])
        resource["fps"] = round(fps)

    if "camera_exif" in patcher_names:
        dataset = kwargs["dataset"]
        index = kwargs["index"]
        dataset.require_configs(dataset.index_columns, "camera_exif_col",
                                f"{dataset.dataset_tag}.{dataset.dataset_tag}_index_kwargs.index_columns")
        try:
            camera_exif = json.loads(dataset.index_manager.get_attribute(index, **dataset.index_columns["camera_exif_col"]))
            resource["camera_exif"] = camera_exif
        except Exception as e:
            logger.warning(f"Cannot loading camera exif: {e}")
            resource["camera_exif"] = {}

    return resource


@register_patcher
def image_ratio(prompt: str | tuple[str] | CaptionOut | tuple[CaptionOut], resource: Dict[str, Any], lang: str):
    ratio_index = resource["ratio_index"]
    # 包含中文字符即认为是中文
    ratio_repr = ratio_index_repr(ratio_index, lang)

    def apply(p):
        if p is None:
            return None
        left = random.random() < 0.2
        if left:
            return f"{ratio_repr}{COMMA[lang]()}" + p
        else:
            return p.rstrip(" ,.，。") + f"{COMMA[lang]()}{ratio_repr}{PERIOD[lang]}"

    if ratio_repr:
        if isinstance(prompt, (list, tuple)):
            if isinstance(prompt[0], CaptionOut):
                for p in prompt:
                    p.caption = apply(p.caption)
            else:
                prompt = [apply(p) for p in prompt]
        elif isinstance(prompt, CaptionOut):
            prompt.caption = apply(prompt.caption)
        else:
            prompt = apply(prompt)
    return prompt


@register_patcher
def image_resolution(prompt: str | tuple[str] | CaptionOut | tuple[CaptionOut], resource: Dict[str, Any], lang: str):
    image_width = resource["image_width"]
    image_height = resource["image_height"]
    # 包含中文字符即认为是中文
    resolution_repr = resolution_repr_fn(width=image_width, height=image_height, lang=lang)

    def apply(p):
        if p is None:
            return None
        left = random.random() < 0.2
        if left:
            return f"{resolution_repr}{COMMA[lang]()}" + p
        else:
            return p.rstrip(" ,.，。") + f"{COMMA[lang]()}{resolution_repr}{PERIOD[lang]}"

    if resolution_repr:
        if isinstance(prompt, (list, tuple)):
            if isinstance(prompt[0], CaptionOut):
                for p in prompt:
                    p.caption = apply(p.caption)
            else:
                prompt = [apply(p) for p in prompt]
        elif isinstance(prompt, CaptionOut):
            prompt.caption = apply(prompt.caption)
        else:
            prompt = apply(prompt)
    return prompt

@register_patcher
def fps(prompt: str | CaptionOut, resource: Dict[str, Any], lang: str):
    fps = resource["fps"]
    # Here we always use english for fps patcher, avoiding detecting the language of the prompt.
    if isinstance(prompt, str):
        return f"FPS: {fps}, {prompt}"
    elif isinstance(prompt, CaptionOut):
        prompt.caption = f"FPS: {fps}, {prompt.caption}"
    else:
        raise ValueError(f"Unsupported prompt type for fps patcher: {type(prompt)}")
    return prompt


def _is_valid_exif_value(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        value = value.strip()
        return value != "" and value.lower() not in {"none", "null", "nan", "unknown", "n/a"}
    return True


def _to_float(value):
    if not _is_valid_exif_value(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _compact_number(value):
    if value is None:
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_exposure(value, lang):
    exposure = _to_float(value)
    if exposure is None or exposure <= 0:
        return None
    if exposure < 1:
        denominator = round(1 / exposure)
        text = f"1/{denominator}s"
    else:
        text = f"{_compact_number(exposure)}s"
    if lang == "zh":
        return random.choice([f"快门速度{text}", f"快门速度: {text}", f"曝光时间{text}", f"曝光时间: {text}", text])
    else:
        return random.choice([f"shutter speed {text}", f"shutter speed: {text}", f"exposure time {text}", f"exposure time: {text}", text])


def _format_aperture(value, lang):
    aperture = _to_float(str(value).lower().replace("f/", "")) if _is_valid_exif_value(value) else None
    if aperture is None or aperture <= 0:
        return None
    text = f"f/{_compact_number(aperture)}"
    if lang == "zh":
        return random.choice([f"光圈{text}", f"光圈: {text}", text])
    else:
        return random.choice([f"aperture {text}", f"aperture: {text}", text])


def _format_focal_length(value, lang):
    focal_length = _to_float(str(value).lower().replace("mm", "")) if _is_valid_exif_value(value) else None
    if focal_length is None or focal_length <= 0:
        return None
    text = f"{_compact_number(focal_length)}mm"
    if lang == "zh":
        return random.choice([f"焦距{text}", f"焦距: {text}", text])
    else:
        return random.choice([f"focal length {text}", f"focal length: {text}", text])


def _format_iso(value, lang):
    iso = _to_float(value)
    if iso is None or iso <= 0:
        return None
    text = f"{_compact_number(iso)}"
    if lang == "zh":
        return random.choice([f"ISO{text}", f"iso{text}", f"ISO: {text}", f"iso: {text}", f"ISO：{text}", f"iso：{text}"])
    else:
        return random.choice([f"ISO {text}", f"ISO: {text}", f"ISO: {text}", f"iso: {text}"])


def _format_camera(value, lang):
    if not isinstance(value, dict):
        return None

    make = str(value.get("camera_make", "")).strip() if _is_valid_exif_value(value.get("camera_make")) else ""
    model = str(value.get("camera_model", "")).strip() if _is_valid_exif_value(value.get("camera_model")) else ""
    if make and model and make.lower() in model.lower():
        camera = model
    else:
        camera = " ".join(x for x in [make, model] if x)
    if not camera:
        return None
    if lang == "zh":
        return random.choice([f"{camera}相机", f"{camera}型号", camera])
    else:
        return random.choice([f"{camera} camera", f"{camera} model", camera])


@register_patcher
def camera_exif(prompt: str | CaptionOut, resource: Dict[str, Any], lang: str):
    exif = resource.get("camera_exif", {})
    if not isinstance(exif, dict):
        return prompt

    fields = [
        _format_camera(exif, lang),
        _format_exposure(exif.get("exposure_time"), lang),
        _format_aperture(exif.get("aperture_fnumber"), lang),
        _format_focal_length(exif.get("focal_length_mm"), lang),
        _format_iso(exif.get("iso"), lang),
    ]
    fields = [field for field in fields if field]
    if not fields:
        return prompt

    if lang == "zh":
        prefix = random.choice([f"拍摄参数：", f"拍摄参数: ", f"拍摄参数: ", f"拍摄参数: ", ""])
        connector = random.choice([f"，", f"。", " "])
    else:
        prefix = random.choice([f"Camera settings: ", f"Camera settings: ", f"Camera settings: ", f"Camera settings: ", ""])
        connector = random.choice([f", ", f",", " "])
    metadata = prefix + (connector.join(fields))

    def apply(p):
        if p is None:
            return None
        if lang == "zh":
            return random.choice([f"{metadata}。{p}", f"{p.rstrip(" ,.，。")}。{metadata}。", f"{p.rstrip(" ,.，。")}，{metadata}。"])
        else:
            return random.choice([f"{metadata}. {p}", f"{p.rstrip(" ,.，。")}. {metadata}.", f"{p.rstrip(" ,.，。")}, {metadata}."])

    if isinstance(prompt, (list, tuple)):
        if prompt and isinstance(prompt[0], CaptionOut):
            for p in prompt:
                p.caption = apply(p.caption)
        else:
            prompt = [apply(p) for p in prompt]
    elif isinstance(prompt, CaptionOut):
        prompt.caption = apply(prompt.caption)
    else:
        prompt = apply(prompt)
    return prompt


def apply_prompt_patchers(prompt, patcher_names, lang=None, **kwargs):
    invalid_names = []
    for name in patcher_names:
        name = name.split('@')[0]   # Remove prob suffix
        if name not in PROMPT_PATCHERS:
            invalid_names.append(name)
    if invalid_names:
        raise ValueError(
            f"Invalid patcher names: {', '.join(invalid_names)}. "
            f"Available patchers: {', '.join(PROMPT_PATCHERS.keys())}"
        )

    simple_patcher_names = [name.split('@')[0] for name in patcher_names]
    resource = load_resource(patcher_names=simple_patcher_names, **kwargs)
    # lang = "zh" if any('\u4e00' <= c <= '\u9fff' for c in prompt) else "en"
    for name in patcher_names:
        if '@' in name:
            name, prob = name.split('@')
            prob = float(prob)
        else:
            prob = 1.0
        patcher = PROMPT_PATCHERS[name]
        if prob == 1.0 or random.random() < float(prob):
            is_tuple = isinstance(prompt, tuple)
            prompt = patcher(prompt, resource=resource, lang=lang)
            if is_tuple:
                prompt = tuple(prompt)
    return prompt
