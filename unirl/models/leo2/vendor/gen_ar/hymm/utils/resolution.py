import numpy as np
import random

RESOLUTIONS = {
    # ID:        height width              h/16 w/16           tokens          ratio            repr
     0: {"size": ( 512, 2048), "tk_size": ( 32, 128), "n_token": 4096, "ratio": 0.2500, "repr": {"common": ["2048x512", "4:1"]}},
     1: {"size": ( 512, 1984), "tk_size": ( 32, 124), "n_token": 3968, "ratio": 0.2581, "repr": {"common": ["1984x512", "3.875:1"]}},
     2: {"size": ( 512, 1920), "tk_size": ( 32, 120), "n_token": 3840, "ratio": 0.2667, "repr": {"common": ["1920x512", "3.75:1"]}},
     3: {"size": ( 512, 1856), "tk_size": ( 32, 116), "n_token": 3712, "ratio": 0.2759, "repr": {"common": ["1856x512", "3.625:1"]}},
     4: {"size": ( 512, 1792), "tk_size": ( 32, 112), "n_token": 3584, "ratio": 0.2857, "repr": {"common": ["1792x512", "3.5:1"]}},
     5: {"size": ( 512, 1728), "tk_size": ( 32, 108), "n_token": 3456, "ratio": 0.2963, "repr": {"common": ["1728x512", "3.375:1"]}},
     6: {"size": ( 512, 1664), "tk_size": ( 32, 104), "n_token": 3328, "ratio": 0.3077, "repr": {"common": ["1664x512", "3.25:1"]}},
     7: {"size": ( 512, 1600), "tk_size": ( 32, 100), "n_token": 3200, "ratio": 0.3200, "repr": {"common": ["1600x512", "3.125:1"]}},
     8: {"size": ( 512, 1536), "tk_size": ( 32,  96), "n_token": 3072, "ratio": 0.3333, "repr": {"common": ["1536x512", "3:1"]}},
     9: {"size": ( 576, 1472), "tk_size": ( 36,  92), "n_token": 3312, "ratio": 0.3913, "repr": {"common": ["1472x576", ]}},
    10: {"size": ( 640, 1408), "tk_size": ( 40,  88), "n_token": 3520, "ratio": 0.4545, "repr": {
        "common": ["1408x640", "21:9", "21:9", "21:9", "2.35:1", "2.35:1"],
        "zh": ["超宽屏", "影院宽屏", "电影比例", "电影画幅", "沉浸式宽屏"],
        "en": ["ultra wide", "cinemascope", "panoramic", "horizontal"],
    }},
    11: {"size": ( 704, 1344), "tk_size": ( 44,  84), "n_token": 3696, "ratio": 0.5238, "repr": {
        "common": ["1344x704", "2:1", "2:1", "2:1"],
        "zh": ["超宽屏", "双倍宽屏", "双倍宽度"],
        "en": ["ultra wide", "double width", "horizontal"],
    }},
    12: {"size": ( 768, 1280), "tk_size": ( 48,  80), "n_token": 3840, "ratio": 0.6000, "repr": {
        "common": ["1280x768", "16:9", "16:9", "16:9"],
        "zh": ["宽屏", "宽屏", "宽屏", "宽屏", "宽屏", "横屏", "横屏", "横屏", "高清比例", "全高清"],
        "en": ["widescreen", "HDTV", "full HD", "horizontal", "landscape"],
    }},
    13: {"size": ( 832, 1216), "tk_size": ( 52,  76), "n_token": 3952, "ratio": 0.6842, "repr": {
        "common": ["1216x832", "3:2", "3:2", "3:2"],
        "zh": ["宽屏", "宽屏", "宽屏", "横屏", "横屏", "横屏", "微单比例", "摄影画幅", "相机画幅", "明信片比例"],
        "en": ["widescreen", "camera aspect", "postcard ratio", "horizontal"],
    }},
    14: {"size": ( 896, 1152), "tk_size": ( 56,  72), "n_token": 4032, "ratio": 0.7778, "repr": {
        "common": ["1152x896", "4:3", "4:3", "4:3", "5:4", "5:4", "5:4"],
        "zh": ["宽屏", "宽屏", "宽屏", "横屏", "横屏", "横屏", "经典屏幕", "传统比例"],
        "en": ["widescreen", "CRT ratio", "horizontal"],
    }},
    15: {"size": ( 960, 1088), "tk_size": ( 60,  68), "n_token": 4080, "ratio": 0.8824, "repr": {
        "common": ["1088x960", ],
        "zh": [],
        "en": ["near square", "horizontal"],
    }},
    16: {"size": (1024, 1024), "tk_size": ( 64,  64), "n_token": 4096, "ratio": 1.0000, "repr": {
        "common": ["1024x1024", "1:1", "1:1", "1:1"],
        "zh": ["方屏", "方形", "正方形", "头像比例"],
        "en": ["square", "profile pic"],
    }},
    17: {"size": (1088,  960), "tk_size": ( 68,  60), "n_token": 4080, "ratio": 1.1333, "repr": {
        "common": ["960x1088", ],
        "zh": [],
        "en": ["vertical"],
    }},
    18: {"size": (1152,  896), "tk_size": ( 72,  56), "n_token": 4032, "ratio": 1.2857, "repr": {
        "common": ["896x1152", "3:4", "3:4", "4:5", "4:5"],
        "zh": ["竖屏", "竖屏"],
        "en": ["vertical", "portrait"],
    }},
    19: {"size": (1216,  832), "tk_size": ( 76,  52), "n_token": 3952, "ratio": 1.4615, "repr": {
        "common": ["832x1216", "2:3", "2:3"],
        "zh": ["竖屏", "竖屏"],
        "en": ["photo print", "poster portrait", "vertical"],
    }},
    20: {"size": (1280,  768), "tk_size": ( 80,  48), "n_token": 3840, "ratio": 1.6667, "repr": {
        "common": ["768x1280", "9:16", "9:16"],
        "zh": ["竖屏", "竖屏"],
        "en": ["vertical"],
    }},
    21: {"size": (1344,  704), "tk_size": ( 84,  44), "n_token": 3696, "ratio": 1.9091, "repr": {
        "common": ["704x1344", "1:2", "1:2"],
        "zh": ["超窄屏", "竖长图"],
        "en": ["vertical"],
    }},
    22: {"size": (1408,  640), "tk_size": ( 88,  40), "n_token": 3520, "ratio": 2.2000, "repr": {
        "common": ["640x1408", "9:21", "9:21"],
        "zh": ["超窄屏", "竖长图"],
        "en": ["vertical"],
    }},
    23: {"size": (1472,  576), "tk_size": ( 92,  36), "n_token": 3312, "ratio": 2.5556, "repr": {"common": ["576x1472", ]}},
    24: {"size": (1536,  512), "tk_size": ( 96,  32), "n_token": 3072, "ratio": 3.0000, "repr": {"common": ["512x1536", "1:3"]}},
    25: {"size": (1600,  512), "tk_size": (100,  32), "n_token": 3200, "ratio": 3.1250, "repr": {"common": ["512x1600", "1:3.125"]}},
    26: {"size": (1664,  512), "tk_size": (104,  32), "n_token": 3328, "ratio": 3.2500, "repr": {"common": ["512x1664", "1:3.25"]}},
    27: {"size": (1728,  512), "tk_size": (108,  32), "n_token": 3456, "ratio": 3.3750, "repr": {"common": ["512x1728", "1:3.375"]}},
    28: {"size": (1792,  512), "tk_size": (112,  32), "n_token": 3584, "ratio": 3.5000, "repr": {"common": ["512x1792", "1:3.5"]}},
    29: {"size": (1856,  512), "tk_size": (116,  32), "n_token": 3712, "ratio": 3.6250, "repr": {"common": ["512x1856", "1:3.625"]}},
    30: {"size": (1920,  512), "tk_size": (120,  32), "n_token": 3840, "ratio": 3.7500, "repr": {"common": ["512x1920", "1:3.75"]}},
    31: {"size": (1984,  512), "tk_size": (124,  32), "n_token": 3968, "ratio": 3.8750, "repr": {"common": ["512x1984", "1:3.875"]}},
    32: {"size": (2048,  512), "tk_size": (128,  32), "n_token": 4096, "ratio": 4.0000, "repr": {"common": ["512x2048", "1:4"]}},
    33: {"size": (1024,  768), "tk_size": (64,   48), "n_token": 3072, "ratio": 1.3333, "repr": {"common": ["768x1024", "3:4"]}},
    34: {"size": (1280,  720), "tk_size": (80,   45), "n_token": 3600, "ratio": 1.7778, "repr": {"common": ["720x1280", "9:16"]}},
    35: {"size": (768,  1024), "tk_size": (48,   64), "n_token": 3072, "ratio": 0.7500, "repr": {"common": ["1024x768", "4:3"]}},
    36: {"size": (720,  1280), "tk_size": (45,   80), "n_token": 3600, "ratio": 0.5625, "repr": {"common": ["1280x720", "16:9"]}},

}

def ratio_index_repr(ratio_index, lang):
    reso_repr = RESOLUTIONS[ratio_index]["repr"]
    if lang == "zh":
        candidates = reso_repr["common"] + reso_repr.get("zh", [])
    elif lang == "en":
        candidates = reso_repr["common"] + reso_repr.get("en", [])
    else:
        raise ValueError(f"Unsupported language: {lang}")
    return random.choice(candidates) if len(candidates) > 0 else None

class Resolution(object):
    def __init__(self, size, *args):
        if isinstance(size, str):
            if 'x' in size:
                size = size.split('x')
                size = (int(size[0]), int(size[1]))
            else:
                size = int(size)
        if len(args) > 0:
            size = (size, args[0])
        if isinstance(size, int):
            size = (size, size)

        self.h = self.height = size[0]
        self.w = self.width = size[1]
        self.r = self.ratio = self.height / self.width

    def __getitem__(self, idx):
        if idx == 0:
            return self.h
        elif idx == 1:
            return self.w
        else:
            raise IndexError(f'Index {idx} out of range')

    def __str__(self):
        return f'{self.h}x{self.w}'


class ResolutionGroup(object):
    def __init__(self, base_size=None, step=None, align=1, target_ratios=None, enlarge=1, data=None,
                 num_ratios=None, extra_resolutions=None):
        self.enlarge = enlarge
        self.num_ratios = num_ratios

        if data is not None:
            self.data = data
            mid = len(self.data) // 2
            self.base_size = self.data[mid].h
            self.step = self.data[mid].h - self.data[mid - 1].h
        else:
            self.align = align
            self.base_size = base_size
            assert base_size % align == 0, f'base_size {base_size} is not divisible by align {align}'
            if base_size is not None and not isinstance(base_size, int):
                raise ValueError(f'base_size must be None or int, but got {type(base_size)}')
            if step is None and target_ratios is None:
                step = base_size // 16
            if step is not None and step > base_size // 2:
                raise ValueError(f'step must be smaller than base_size // 2, but got {step} > {base_size // 2}')

            self.step = step
            self.data = self.calc(target_ratios, extra_resolutions)

        self.ratio = np.array([x.ratio for x in self.data])
        if num_ratios is not None:
            assert num_ratios == len(self.ratio), f'num_ratios({num_ratios}) != len(self.ratio)({len(self.ratio)})'
        self.attr = ['' for _ in range(len(self.data))]
        self.prefix_space = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        prefix = self.prefix_space * ' '
        prefix_close = (self.prefix_space - 4) * ' '
        res_str = f'ResolutionGroup(base_size={self.base_size}, step={self.step}, data='
        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += f'\n{prefix}ID: height width   ratio {" " * max(0, attr_maxlen - 4)}count  h/16 w/16    tokens\n{prefix}'
        res_str += ('\n' + prefix).join([f'{i:2d}: ({x.h:4d}, {x.w:4d})  {self.ratio[i]:.4f}  {self.attr[i]:>{attr_maxlen}s}  '
                                         f'({x.h // 16:3d}, {x.w // 16:3d})  {x.h // 16 * x.w // 16:6d}'
                                         for i, x in enumerate(self.data)])
        res_str += f'\n{prefix_close})'
        return res_str

    @staticmethod
    def from_list_of_hxw(hxw_list):
        data = [Resolution(x) for x in hxw_list]
        data = sorted(data, key=lambda x: x.ratio)
        return ResolutionGroup(None, data=data)

    def calc(self, target_ratios=None, extra_resolutions=None):
        if target_ratios is None:
            resolutions = self._calc_by_step()
        else:
            resolutions = self._calc_by_ratio(target_ratios)
        if extra_resolutions is not None:
            for extra_resolution in extra_resolutions:
                height, width = extra_resolution
                ratio = height / width
                flag = True
                for resolution in resolutions:
                    if resolution.ratio == ratio:
                        flag = False
                        break
                if flag:
                    resolutions.append(Resolution(height, width))
                    # do not sort here to keep the order of old resolutions for matching <img_ratio_{i}> tokens

        return resolutions

    def _calc_by_ratio(self, target_ratios):
        resolutions = []
        for ratio in target_ratios:
            if ratio == '1:1':
                reso = Resolution(self.base_size, self.base_size)
            else:
                hr, wr = map(int, ratio.split(':'))
                x = int((self.base_size ** 2 * self.enlarge // self.align // self.align / (hr * wr)) ** 0.5)
                height = x * hr * self.align
                width = x * wr * self.align
                reso = Resolution(height, width)
            resolutions.append(reso)

        resolutions = sorted(resolutions, key=lambda x_: x_.ratio)

        return resolutions

    def _calc_by_step(self):
        assert self.align <= self.step, f'align {self.align} must be smaller than step {self.step}'

        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break

            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break

            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        resolutions = sorted(resolutions, key=lambda x: x.ratio)

        return resolutions

    def get_target_size(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        reso = self.data[idx]
        return reso.w, reso.h

    def get_base_size_and_ratio_index(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        return self.base_size, idx
