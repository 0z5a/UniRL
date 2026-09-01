import random
from loguru import logger

def get_zh_expr_for_a_w_h(width, height):
    return [f"宽{width}高{height}", f"高{height}宽{width}", f"宽{width}，高{height}", f"高{height}，宽{width}", f"{height}高，{width}宽", f"{width}宽，{height}高", f"分辨率{width}x{height}", f"分辨率{width}:{height}", f"分辨率：{width}x{height}", f"分辨率：{width}:{height}"]

def get_en_expr_for_a_w_h(width, height):
    return [f"width: {width}, height: {height}", f"height: {height}, width: {width}", f"{width} width, {height} height", f"{height} height, {width} width", f"resolution: {width}x{height}", f"resolution: {width}:{height}"]

RESOLUTIONS = {}
for i in range(1, 513):
    for j in range(1, 513):
        width = i*16
        height = j*16
        ratio = width / height
        extra_zh = []
        extra_en = []
        if ratio >= 9/20 and ratio <= 3/4:
            extra_zh.append("竖屏")
            extra_en.append("vertical")
        if ratio >= 4/3 and ratio <= 21/9:
            extra_zh.append("横屏")
            extra_en.append("horizontal")

        RESOLUTIONS[f"{width}x{height}"] = {
            "width": width,
            "height": height,
            "repr": {
                "common": [f"{width}x{height}", f"{width}:{height}"],
                "zh": get_zh_expr_for_a_w_h(width=width, height=height) + extra_zh,
                "en": get_en_expr_for_a_w_h(width=width, height=height) + extra_en,
            },
        }

SPECIAL_RESOLUTIONS = {
    "3024x1296": {
        "common": ["21:9", "21:9", "21:9"],
        "zh": ["超宽屏", "超宽屏", "电影荧幕比例", "影院宽屏", "电影比例", "电影画幅"],
        "en": ["ultra wide", "cinemascope", "panoramic", "cinemascope", "panoramic", "horizontal"],
    },
    "2560x1440": {
        "common": ["16:9", "16:9", "16:9", "3840x2160"],
        "zh": ["1440p", "1440P", "2K", "2K宽屏", "2K比例", "2K横屏"] + get_zh_expr_for_a_w_h(width=3840, height=2160),
        "en": ["1440p", "1440P", "2K", "2K wide", "2K aspect", "2K horizontal"] + get_en_expr_for_a_w_h(width=3840, height=2160),
    },
    "2496x1664": {
        "common": ["3:2", "3:2", "3:2", "3000x2000"],
        "zh": ["全画幅", "全画幅相机比例"] + get_zh_expr_for_a_w_h(width=3000, height=2000),
        "en": ["full frame", "full frame camera aspect"] + get_en_expr_for_a_w_h(width=3000, height=2000),
    },
    "2304x1728": {
        "common": ["4:3", "4:3", "4:3", "3072x2304"],
        "zh": ["中画幅", "中画幅相机比例", "iPad 屏幕比例"] + get_zh_expr_for_a_w_h(width=3072, height=2304),
        "en": ["medium format", "medium format camera aspect", "iPad aspect"] + get_en_expr_for_a_w_h(width=3072, height=2304),
    },
    "2048x2048": {
        "common": ["1:1", "1:1", "1:1"],
        "zh": ["方屏", "方形", "正方形", "方图"],
        "en": ["square", "square image", "square picture"],
    },
    "1728x2304": {
        "common": ["3:4", "3:4", "3:4", "2304x3072"],
        "zh": [] + get_zh_expr_for_a_w_h(width=2304, height=3072),
        "en": [] + get_en_expr_for_a_w_h(width=2304, height=3072),
    },
    "1664x2496": {
        "common": ["2:3", "2:3", "2:3", "2000x3000"],
        "zh": [] + get_zh_expr_for_a_w_h(width=2000, height=3000),
        "en": [] + get_en_expr_for_a_w_h(width=2000, height=3000),
    },
    "1440x2560": {
        "common": ["9:16", "9:16", "9:16", "2160x3840"],
        "zh": [] + get_zh_expr_for_a_w_h(width=2160, height=3840),
        "en": [] + get_en_expr_for_a_w_h(width=2160, height=3840),
    }
}

for key, value in SPECIAL_RESOLUTIONS.items():
    RESOLUTIONS[key]["repr"]["common"] = RESOLUTIONS[key]["repr"]["common"] + value["common"]
    RESOLUTIONS[key]["repr"]["zh"] = RESOLUTIONS[key]["repr"]["zh"] + value["zh"]
    RESOLUTIONS[key]["repr"]["en"] = RESOLUTIONS[key]["repr"]["en"] + value["en"]



def resolution_repr_fn(width, height, lang):
    key = f"{width}x{height}"
    if key not in RESOLUTIONS:
        logger.warning(f"Resolution {key} not found in RESOLUTIONS")
    resolution_repr = RESOLUTIONS[key]["repr"]
    if lang == "zh":
        candidates = [random.choice(["{x}", "比例：{x}", "尺寸比例：{x}", "比例是{x}", "比例为{x}", "图片比例为{x}", "图像比例为{x}", "图片宽高比是{x}", "图片宽高比为{x}", "图片长宽比是{x}", "图片长宽比为{x}", "宽高比为{x}", "长宽比为{x}", "一张比例为{x}的图片", "具有{x}宽高比的图像", "具有{x}长宽比的图像"]).format(x=i) for i in resolution_repr["common"]] + resolution_repr.get("zh", [])
    elif lang == "en":
        candidates = [random.choice(["{x}", "ratio: {x}", "aspect ratio: {x}", "the ratio is {x}", "the image ratio is {x}", "the aspect ratio is {x}", "the image aspect ratio is {x}", "an image with a ratio of {x}", "an image with an aspect ratio of {x}", "an image with {x} aspect ratio", "a picture with a ratio of {x}", "a picture with an aspect ratio of {x}", "a picture with {x} aspect ratio"]).format(x=i) for i in resolution_repr["common"]] + resolution_repr.get("en", [])
    else:
        raise ValueError(f"Unsupported language: {lang}")
    return random.choice(candidates) if len(candidates) > 0 else None

