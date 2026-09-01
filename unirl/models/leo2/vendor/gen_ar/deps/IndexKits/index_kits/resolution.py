import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve


class Resolution(object):
    def __init__(self, size: int | str | tuple, *args, ar=""):
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
        self.size = (self.w, self.h)
        self.ar = ar
        if self.ar == "":
            # Determine multipliers
            if self.h / self.w == self.h // self.w:
                multiplier_h = self.h // self.w
                multiplier_w = 1
            elif self.w / self.h == self.w // self.h:
                multiplier_h = 1
                multiplier_w = self.w // self.h
            elif self.w > self.h:
                multiplier_h = 1
                multiplier_w = self.w / self.h
            else:
                multiplier_h = self.h / self.w
                multiplier_w = 1

            if multiplier_h == round(multiplier_h, 3):
                multiplier_h = str(multiplier_h)
            else:
                multiplier_h = f"{multiplier_h:.3f}"
            if multiplier_w == round(multiplier_w, 3):
                multiplier_w = str(multiplier_w)
            else:
                multiplier_w = f"{multiplier_w:.3f}"
            self.ar = f"{multiplier_w:>5s}:{multiplier_h:<5s}"

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
    """
    Define a group of image resolutions.

    Args:
        base_size (int): The base resolution.
        step (int): The step size for generating resolutions.
        align (int): The alignment size.
        mode (str): The mode of resolution generation. Default to sdxl. Supported modes:
            - sdxl: Generate resolutions based on SDXL aspect ratios, including 33 buckets.
            - sdxl+: Generate resolutions based on SDXL mode, including buckets specified by aspect_ratios.
            - arc: Generate resolutions based on points uniformly distributed along the arc length.
            - arc+: Generate resolutions based on arc mode and including buckets specified by aspect_ratios.
            - aspect_ratio: Generate resolutions based on custom aspect ratios.
        num_buckets (int): Number of buckets for arc mode.
        preset (str): Preset mode. If set, mode will be ignored. Supported presets:
            - sdxl: SDXL preset with mode='sdxl', step=base_size//16, align=16.
            - arc33: Arc preset with base_size=1024, num_buckets=33, align=16, mode='arc'.
        aspect_ratios (list of str): List of aspect ratios for custom mode. Each aspect ratio should be in the
            format 'H:W', e.g., '3:4'.
        data (list): Predefined list of Resolution objects.
        added_method (str): Method to add new resolutions. Options are 'insert' or 'append'.
    """
    def __init__(self, base_size: int = None, step: int = None, align: int = 16, mode: str = None, preset: str = None,
                 aspect_ratios: list[str] = None, num_buckets: int = None, data: list[Resolution] = None,
                 added_method="insert", allow_replace=False):
        self.base_size = base_size
        self.step = step
        self.align = align
        self.mode = mode
        self.preset = preset
        self.aspect_ratios = self.parse_aspect_ratio(aspect_ratios)
        self.num_buckets = num_buckets
        self.added_method = added_method
        self.allow_replace = allow_replace

        assert self.added_method in ['insert', 'append'], \
            f"added_method must be selected from [insert, append], got '{self.added_method}' instead."

        if data is not None:
            self.data = data
        else:
            assert self.base_size is not None, 'base_size must be specified if data is not provided'
            assert self.base_size % self.align == 0, f'base_size {self.base_size} is not divisible by align {self.align}'
            if self.base_size is not None and not isinstance(self.base_size, int):
                raise ValueError(f'base_size must be None or int, but got {type(self.base_size)}')

            if self.preset is not None and self.mode is not None:
                raise ValueError('preset and mode cannot be set at the same time')
            if self.preset is not None:
                assert self.preset in ["sdxl", "arc33", "zigzag"], f'preset {self.preset} is not supported'
                if self.step is not None or self.aspect_ratios is not None or self.num_buckets is not None:
                    warnings.warn(
                        f"When preset is set, these parameters will be ignored: step, aspect_ratios, num_buckets."
                    )
                if self.preset == "sdxl":
                    self.mode = "sdxl"
                    self.step = self.base_size // 16
                elif self.preset == "arc33":
                    self.mode = "arc"
                    self.num_buckets = 33
                elif self.preset == "zigzag":
                    self.mode = "zigzag"
                    self.step = 16
            elif self.mode is None:
                self.mode = "sdxl"

            if "sdxl" in self.mode and self.step is None:
                self.step = self.base_size // 16
            if "zigzag" in self.mode and self.step is None:
                self.step = 16

            # mode sanity check
            assert self.mode in ["sdxl", "sdxl+", "arc", "arc+", "aspect_ratio", "zigzag"], f'mode {self.mode} is not supported'
            if self.mode.startswith("arc"):
                assert self.num_buckets is not None, "num_buckets must be specified for arc mode"
            else:
                assert self.num_buckets is None, f"The `{self.mode}` mode does not support num_buckets."
            if self.mode == "aspect_ratio" or "+" in self.mode:
                assert self.aspect_ratios is not None, f"aspect_ratios must be specified for '{self.mode}' mode"
            else:
                assert self.aspect_ratios is None, f"The `{self.mode}` mode does not support aspect_ratios."

            self.data = self.calc(self.mode, self.aspect_ratios, num_buckets=self.num_buckets)

        self.ratio = np.array([x.ratio for x in self.data])
        self.attr = ['' for _ in range(len(self.data))]
        self.prefix_space = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def parse_aspect_ratio(aspect_ratios: list[str]):
        if aspect_ratios is None:
            return aspect_ratios
        if not isinstance(aspect_ratios, list):
            raise TypeError(f'aspect_ratios must be a list, got {type(aspect_ratios)}')

        parsed_aspect_ratios = []
        for ar in aspect_ratios:
            if ar.startswith("(") and ar.endswith(")"):
                ar = ar[1:-1]
                include_reversed = True
            else:
                include_reversed = False
            parsed_aspect_ratios.append(ar)
            if include_reversed:
                # reversed order
                if ':' in ar:
                    h, w = map(int, ar.split(':'))
                    parsed_aspect_ratios.append(f"{w}:{h}")
                elif 'x' in ar:
                    w, h = map(int, ar.split('x'))
                    parsed_aspect_ratios.append(f"{h}x{w}")
                else:
                    raise ValueError(f'Aspect ratio {ar} is not in a valid format, should be like "H:W" or "HxW"')

        # Check range, restricted to (256:1 -> 1:256)
        for ar in parsed_aspect_ratios:
            if ':' in ar:
                w, h = map(float, ar.split(':'))
            elif 'x' in ar:
                h, w = map(float, ar.split('x'))
            else:
                raise ValueError(f'Aspect ratio {ar} is not in a valid format, should be like "H:W" or "HxW"')
            ratio = h / w
            if ratio < 1 / 256 or ratio > 256:
                raise ValueError(f'Aspect ratio {ar} is out of range (1/256 to 256)')

        return parsed_aspect_ratios

    def token_range_and_ratio(self):
        # 计算最短 token 长度和最长 token 长度的比值：
        token_lengths = [x.h // 16 * x.w // 16 for x in self.data]
        min_token_length = min(token_lengths)
        max_token_length = max(token_lengths)
        return f"Token length range: {min_token_length} to {max_token_length} (ratio: {min_token_length / max_token_length:.2f})"

    def __repr__(self):
        prefix = self.prefix_space * ' '
        prefix_close = (self.prefix_space - 4) * ' '
        res_str = f'ResolutionGroup(mode={self.mode}, base_size={self.base_size}'
        if self.mode in ["sdxl", "sdxl+"]:
            res_str += f', step={self.step}'
        elif self.mode in ["arc", "arc+"]:
            res_str += f', num={self.num_buckets}'
        elif self.mode == "aspect_ratio":
            pass
        else:
            res_str = f'ResolutionGroup(mode=unknown, base_size={self.base_size}'
        if str(self.mode).endswith("+"):
            res_str += f', added_method={self.added_method}'
        res_str += ', data='

        ar_maxlen = max([len(x.ar) for x in self.data])
        if ar_maxlen > 0:
            ar_header = " aspect_ratio "
            ar_maxlen = max(ar_maxlen, len(ar_header))
        else:
            ar_header = ""

        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += (f'\n{prefix}ID:  height  width     ratio {ar_header}{" " * max(0, attr_maxlen - 4)}count   h/16  w/16    tokens\n'
                    f'{prefix}')
        res_str += ('\n' + prefix).join([
            f'{i:2d}: ({x.h:5d}, {x.w:5d})  {self.ratio[i]:>8.4f} {x.ar:^{ar_maxlen}} {self.attr[i]:>{attr_maxlen}s}  '
            f'({x.h // 16:4d}, {x.w // 16:4d})  {x.h // 16 * x.w // 16:6d}'
            for i, x in enumerate(self.data)
        ])
        res_str += f'\n{prefix_close})'
        res_str += f"\n{prefix_close}{self.token_range_and_ratio()}"
        return res_str

    def to_df(self, factor=1):
        xs, ys = [], []
        ratios = []
        for reso in self.data:
            xs.append(reso.w // factor)
            ys.append(reso.h // factor)
            ratios.append(reso.ratio)
        df = pd.DataFrame({'x': xs, 'y': ys, 'ratio': ratios})
        return df

    @staticmethod
    def from_list_of_hxw(hxw_list, base_size=None, step=None):
        data = [Resolution(x) for x in hxw_list]
        data = sorted(data, key=lambda x: x.ratio)
        return ResolutionGroup(base_size=base_size, step=step, data=data)

    @staticmethod
    def dedup(data):
        unique_resolutions = {}
        for res in data:
            key = (res.h, res.w)
            if key not in unique_resolutions:
                unique_resolutions[key] = res
        data = list(unique_resolutions.values())
        return data

    def calc(
            self,
            mode: str,
            target_ratios: list[str] = None,
            num_buckets: int = 33,
    ) -> list[Resolution]:
        if mode in ["sdxl", "sdxl+"]:
            data = self._calc_by_step()
        elif mode in ["arc", "arc+"]:
            data = self._calc_by_arc(n=num_buckets)
        elif mode == "aspect_ratio":
            data = self._calc_by_ratio(target_ratios)
        elif mode == "zigzag":
            data = self._calc_by_step_zigzag()
        else:
            raise ValueError(f'mode {mode} is not supported')

        if self.added_method == "append":
            data = sorted(data, key=lambda x: x.ratio)

        if "+" in mode:
            extra_data = self._calc_by_ratio(target_ratios)
            # Remove existing ratios if extra_data has the same aspect ratio
            extra_ratios = set(res.ratio for res in extra_data)
            ratios = set(res.ratio for res in data)
            if not self.allow_replace:
                assert len(extra_ratios.intersection(ratios)) == 0, \
                    (f"Added aspect ratios {extra_ratios.intersection(ratios)} already exist in the original data. "
                     f"Set allow_replace=True to allow replacement.")
            data = [res for res in data if res.ratio not in extra_ratios]
            data = data + extra_data

        data = self.dedup(data)

        if self.added_method == "insert":
            data = sorted(data, key=lambda x: x.ratio)

        return data

    def _calc_by_step(self) -> list[Resolution]:
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

        return resolutions

    def _calc_by_step_zigzag(self, max_ratio=4.0) -> list[Resolution]:
        assert self.align <= self.step, f'align {self.align} must be smaller than step {self.step}'
        num_patches = round((self.base_size / self.step) ** 2)
        resolutions = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                resolutions.append((wp * self.step // self.align * self.align, hp * self.step // self.align * self.align))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        resolutions = [Resolution(x) for x in resolutions]
        return resolutions

    def _calc_by_arc(self, n=33) -> list[Resolution]:
        """
        Generate resolutions based on points uniformly distributed along the arc length.

        Define an arc from [a, b] to [a, b] with x*y = r.
        """
        assert n % 2 == 1, f'n {n} must be odd'

        a = self.base_size // 2 // self.align
        b = self.base_size * 2 // self.align

        # 定义被积函数
        def integrand(u):
            return np.sqrt(np.cosh(2 * u))

        # 定义计算 L 的函数
        def compute_L(a, b, t):
            # 计算从 0 到 t 的积分
            integral_result, _ = quad(integrand, 0, t)
            # 计算最终的 L
            L = np.sqrt(2 * a * b) * 2 * integral_result
            return L

        # 定义积分计算函数
        def integral(t):
            result, _ = quad(integrand, 0, t)
            return result

        # 定义目标函数，求解方程 integral(t) = S
        def equation(t, S):
            return integral(t) - S

        # 首先计算弧长
        t0 = 0.5 * np.log(b / a)
        arc_len = compute_L(a, b, t0)

        # 计算每个分段的弧长
        si = arc_len / (n - 1)

        # 求解每个分段的 t 值
        half_ts = []
        for i in range(1, n // 2):
            S = si * i / (np.sqrt(2 * a * b))
            initial_guess = 1  # 初始猜测值
            t_solution = fsolve(equation, initial_guess, args=(S,))
            half_ts.extend(t_solution)
        ts = [t0] + half_ts[::-1] + [0.0] + [-t for t in half_ts] + [-t0]

        # 计算对应的分辨率
        resolutions = []
        for t in ts:
            x = np.sqrt(a * b) * np.exp(t)
            y = np.sqrt(a * b) * np.exp(-t)
            resolutions.append(Resolution(int(y) * self.align, int(x) * self.align))

        return resolutions

    def _calc_by_ratio(self, target_ratios: list[str]) -> list[Resolution]:
        # 初始化最大面积和对应的 h, w
        max_area = np.zeros((len(target_ratios),), np.int64)
        best_h = np.zeros((len(target_ratios),), np.int64)
        best_w = np.zeros((len(target_ratios),), np.int64)
        if ':' in target_ratios[0]:
            assert all(':' in r for r in target_ratios), "All aspect ratios should be in 'H:W' format if ':' is used."
            sizes = [list(map(int, r.split(':'))) for r in target_ratios]
            ratios = np.array([h / w for w, h in sizes])
        elif 'x' in target_ratios[0]:
            assert all('x' in r for r in target_ratios), "All aspect ratios should be in 'HxW' format if 'x' is used."
            sizes = [list(map(int, r.split('x'))) for r in target_ratios]
            assert all(sz[0] % self.align == 0 and sz[1] % self.align == 0 for sz in sizes), \
                "All aspect ratios should have dimensions that are multiples of align."
            resolutions = []
            for h, w in sizes:
                if h <= 0 or w <= 0:
                    raise ValueError(f"Aspect ratio dimensions must be positive, got {h}x{w}.")
                resolutions.append(Resolution(h, w, ar=f"{h:>5d}x{w:<5d}"))
            return resolutions
        else:
            raise ValueError("Aspect ratios should be in 'H:W' or 'HxW' format.")

        # We set a boundary of (base_size/8, base_size*8) to search for the best w
        a = max(self.base_size // 16 // self.align, 1)
        b = min(self.base_size * 16 // self.align, self.base_size ** 2 // self.align // self.align)
        S2 = self.base_size ** 2 // self.align // self.align

        for w in range(a, b + 1):
            hs = ratios * w

            for i, h in enumerate(hs):
                if h != int(h):
                    continue

                area = h * w
                if area <= S2 and area > max_area[i]:
                    max_area[i] = area
                    best_h[i] = int(h) * self.align
                    best_w[i] = int(w) * self.align

        resolutions = []
        for height, width, (hr, wr) in zip(best_h, best_w, sizes):
            resolutions.append(Resolution(int(height), int(width), ar=f"{hr:>5d}:{wr:<5d}"))

        return resolutions

    def get_target_size(self, width: int, height: int) -> tuple[int, int]:
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        reso = self.data[idx]
        return reso.w, reso.h

    def get_base_size_and_ratio_index(self, width: int, height: int) -> tuple[int, int]:
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        return self.base_size, int(idx)


class DurationGroup(object):
    def __init__(self, duration_range: tuple[int, int], duration_step: int, additional_durations: list[int] = None):
        self.duration_range = duration_range
        self.duration_step = duration_step
        self.additional_durations = additional_durations
        data = list(range(self.duration_range[0], self.duration_range[1] + 1, self.duration_step))
        # Include boundary values
        if data[-1] != self.duration_range[1]:
            data.append(self.duration_range[1])
        # Include additional durations
        if self.additional_durations is not None:
            for dur in self.additional_durations:
                if dur not in data:
                    data.append(dur)
        data = sorted(data)
        self.data = np.array(data, dtype=np.int32)

    def __repr__(self):
        return f'DurationGroup(num_buckets={len(self.data)}, durations={self.data.tolist()})'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DurationAndResolution(object):
    def __init__(self, duration: int, size: int | str | tuple, *args, ar=""):
        self.duration = duration
        self.resolution = Resolution(size, *args, ar=ar)

        self.f = self.duration
        self.h = self.resolution.h
        self.w = self.resolution.w
        self.r = self.resolution.r
        self.size = (self.w, self.h, self.f)

    def __getitem__(self, idx):
        if idx == 0:
            return self.f
        elif idx == 1:
            return self.h
        elif idx == 2:
            return self.w
        else:
            raise IndexError(f'Index {idx} out of range')

    def __str__(self):
        return f'{self.f}x{self.h}x{self.w}'


class DurationAndResolutionGroup(object):
    def __init__(
            self,
            duration_range: tuple[int, int],
            duration_step: int,
            additional_durations: list[int] = None,
            base_size: int = None,
            step: int = None,
            align: int = 16,
            mode: str = None,
            preset: str = None,
            aspect_ratios: list[str] = None,
            num_buckets: int = None,
            data: list[Resolution] = None,
            added_method="insert",
    ):
        self.duration_group = DurationGroup(
            duration_range=duration_range,
            duration_step=duration_step,
            additional_durations=additional_durations,
        )
        self.resolution_group = ResolutionGroup(
            base_size=base_size,
            step=step,
            align=align,
            mode=mode,
            preset=preset,
            aspect_ratios=aspect_ratios,
            num_buckets=num_buckets,
            data=data,
            added_method=added_method,
        )
        self.resolution_group.prefix_space = 8

        self.duration_start = self.duration_group.data[0]
        self.duration_end = self.duration_group.data[-1]
        self.duration_step = duration_step
        self.additional_durations = additional_durations

        self.reso_base_size = self.resolution_group.base_size
        self.reso_step = self.resolution_group.step
        self.reso_align = self.resolution_group.align
        self.reso_mode = self.resolution_group.mode
        self.reso_preset = self.resolution_group.preset
        self.reso_aspect_ratios = self.resolution_group.aspect_ratios
        self.reso_num_buckets = self.resolution_group.num_buckets
        self.reso_added_method = self.resolution_group.added_method

        # Cross product
        self.data = []
        for duration in self.duration_group.data:
            for reso in self.resolution_group.data:
                self.data.append(DurationAndResolution(duration, (reso.h, reso.w), ar=reso.ar))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        res_str = (
            f"DurationAndResolutionGroup(\n"
            f"    duration_group={self.duration_group},\n"
            f"    resolution_group={self.resolution_group}\n"
            f")"
        )
        return res_str

    def get_target_size(self, width: int, height: int, duration: int) -> tuple[int, int, int]:
        # Find closest resolution
        ratio = height / width
        reso_idx = np.argmin(np.abs(self.resolution_group.ratio - ratio))
        reso = self.resolution_group.data[reso_idx]

        # Find closest bucket duration <= target duration
        lower_duration = self.duration_group.data[0]
        for candidate_duration in self.duration_group.data[1:]:
            if duration - candidate_duration >= 0:
                lower_duration = candidate_duration

        return reso.w, reso.h, lower_duration

    def get_base_size_and_ratio_index(self, width: int, height: int, duration: int) -> tuple[int, int, int]:
        w, h, f = self.get_target_size(width, height, duration)

        ratio = h / w
        reso_idx = np.argmin(np.abs(self.resolution_group.ratio - ratio))

        duration_idx = np.where(self.duration_group.data == f)[0][0]

        return self.resolution_group.base_size, int(reso_idx), int(duration_idx)
