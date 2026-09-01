# =======================================================
# Caption process script used for caption_v4_zh and
# caption_v4_en
# =======================================================

import json
import random
import re

from .caption_base import CaptionOut

SEPARATORS = {
    "zh": {"comma": "，", "period": "。"},
    "en": {"comma": ", ", "period": ". "}
}


class CameraMovement:

    class_to_desc = {
        '360度': {
            "zh": ["相机进行360度旋转", "镜头完成一圈旋转", "相机环绕一周", "360度旋转运镜"],
            "en": ["The camera rotates 360 degrees", "The camera makes a full rotation"]
        },
        'FPV第一人称视角': {
            "zh": ["相机采用第一人称视角拍摄", "画面以第一人称视角呈现", "镜头为FPV视角", "第一人称视角"],
            "en": ["The camera shows a first-person view", "The camera adopts an FPV angle", "The camera is in a first-person view"]
        },
        '上升': {
            "zh": ["相机向上移动", "镜头上升", "相机升高拍摄"],
            "en": ["The camera moves upward", "The camera rises", "The camera ascends"]
        },
        '下降': {
            "zh": ["相机向下移动", "镜头下降", "相机降低拍摄"],
            "en": ["The camera moves downward", "The camera descends", "The camera drops down"]
        },
        '低角度/仰拍': {
            "zh": ["相机以低角度仰拍", "低角度仰拍"],
            "en": ["The camera shoots from a low angle", "Low angle shot"]
        },
        '倾斜_上': {
            "zh": ["相机向上倾斜", "相机进行上仰拍摄"],
            "en": ["The camera tilts up", "The camera angle shifts upward", "The camera moves in an upward tilt"]
        },
        '倾斜_下': {
            "zh": ["相机向下倾斜", "相机进行下俯拍摄"],
            "en": ["The camera tilts down", "The camera angle shifts downward", "The camera moves in a downward tilt"]
        },
        '地面镜头': {
            "zh": ["相机位于地面高度", "地面镜头"],
            "en": ["The camera is at ground level", "The camera shoots from the ground", "Ground level shot"]
        },
        '平移_前': {
            "zh": ["相机向前移动", "相机向前推近", "相机推近", "镜头推近"],
            "en": ["The camera moves forward", "The camera dollies forward", "The camera dollies in", "The camera pushes forward", "Dolly in"]
        },
        '平移_后': {
            "zh": ["相机向后移动", "相机向后拉远", "相机拉远", "镜头拉远"],
            "en": ["The camera moves backward", "The camera dollies backward", "The camera dollies out", "The camera pulls back", "Dolly out"]
        },
        '平移_右': {
            "zh": ["相机向右平移", "镜头向右移动", "相机右移拍摄"],
            "en": ["The camera moves to the right", "The camera shifts rightward", "The camera trucks right"]
        },
        '平移_左': {
            "zh": ["相机向左平移", "镜头向左移动", "相机左移拍摄"],
            "en": ["The camera moves to the left", "The camera shifts leftward", "The camera trucks left"]
        },
        '延时摄影': {
            "zh": ["相机进行延时摄影", "画面采用延时拍摄", "场景以延时方式呈现", "延时摄影"],
            "en": ["The camera captures a time-lapse", "A time-lapse shot is used", "The scene is shown in time-lapse"]
        },
        '微距': {
            "zh": ["相机进行微距拍摄", "采用微距视角", "微距拍摄", "微距视角", "微距镜头"],
            "en": ["The camera takes a macro shot", "A macro perspective is used"]
        },
        '慢镜头': {
            "zh": ["相机以慢动作拍摄", "画面呈现慢镜头效果", "场景以慢动作捕捉", "慢镜头"],
            "en": ["The camera records in slow motion", "A slow-motion shot is shown", "The scene is captured in slow motion"]
        },
        '拉远': {
            "zh": ["相机变焦拉远"],
            "en": ["The camera zooms out", "Zoom out shot", "Zoom out"]
        },
        '推进': {
            "zh": ["相机变焦拉近"],
            "en": ["The camera zooms in", "Zoom in shot", "Zoom in"]
        },
        '摇_右': {
            "zh": ["相机向右摇动", "相机向右摇摄", "右向摇摄", "右摇运镜"],
            "en": ["The camera pans to the right", "The camera pans right"]
        },
        '摇_左': {
            "zh": ["相机向左摇动", "相机向左摇摄", "左向摇摄", "左摇运镜"],
            "en": ["The camera pans to the left", "The camera pans left"]
        },
        '无人机视角': {
            "zh": ["相机采用无人机视角", "画面以无人机俯拍", "无人机视角", "无人机俯拍"],
            "en": ["The camera shows a drone view", "Drone view", "Aerial shot"]
        },
        '环绕': {
            "zh": ["相机环绕主体拍摄", "相机围绕主体旋转", "环绕拍摄", "环绕运镜"],
            "en": ["The camera circles around the subject", "The camera orbits the subject", "Orbit shot"]
        },
        '跟随镜头': {
            "zh": ["相机跟随主体移动", "镜头追踪运动", "相机与主体同步移动", "跟随拍摄", "跟随运镜", "跟随镜头"],
            "en": ["The camera follows the subject", "The camera tracks the movement", "The camera moves along with the subject"]
        },
        '过肩镜头': {
            "zh": ["相机采用过肩镜头", "画面从人物肩后拍摄", "相机位于主体肩膀后方", "过肩拍摄", "过肩视角", "过肩镜头"],
            "en": ["The camera uses an over-the-shoulder shot", "The scene is shown from over the shoulder", "The camera is in OTS perspective"]
        },
        '逆时针滚转': {
            "zh": ["相机逆时针滚转", "镜头向左旋转", "相机进行逆时针方向旋转"],
            "en": ["The camera rolls counterclockwise", "The camera rotates to the left", "The camera spins in a counterclockwise direction"]
        },
        '静止': {
            # 增加一个空描述
            "zh": ["相机保持静止", "镜头不移动", "相机处于静止状态", "静止拍摄", "相机静止", "静止镜头", ""],
            "en": ["The camera remains still", "The camera is stationary", "The camera does not move", "The camera remains static", "Static shot", ""]
        },
        '顺时针滚转': {
            "zh": ["相机顺时针滚转", "镜头向右旋转", "相机进行顺时针方向旋转"],
            "en": ["The camera rolls clockwise", "The camera rotates to the right", "The camera spins in a clockwise direction"]
        },
        '高角度/俯拍': {
            "zh": ["相机以高角度俯拍", "镜头从上方捕捉画面", "高角度俯拍"],
            "en": ["The camera shoots from a high angle", "The camera captures the scene from above", "High angle shot"]
        },
        '鱼眼镜头': {
            "zh": ["相机采用鱼眼镜头", "画面呈现鱼眼效果", "使用鱼眼视角拍摄"],
            "en": ["The camera uses a fisheye lens", "The scene is shown with a fisheye effect", "A fisheye perspective is applied", "Fisheye lens"]
        },
        '鸟瞰/头顶': {
            "zh": ["相机以鸟瞰视角拍摄", "画面从上方俯视", "鸟瞰视角", "鸟瞰镜头"],
            "en": ["The camera shows a bird's-eye view", "The scene is captured from overhead", "Overhead view"]
        },

        # 特殊运镜
        '低角度跟随': {
            "zh": ["相机以低角度跟随拍摄", "低角度跟随", "低角度跟随镜头"],
            "en": ["The camera follows from a low angle", "Low-angle tracking shot"]
        },
        '头顶越肩_撞击式推近': {
            "zh": ["相机由头顶越肩，撞击式推近主体"],
            "en": ["The camera pushes in with an overhead over-the-shoulder shot"]
        },
        '子弹时间': {
            "zh": ["相机环绕主体营造子弹时间效果", "子弹时间特效镜头", "子弹时间"],
            "en": ["Bullet time effect", "Bullet time shot"]
        },
        '希区柯克': {
            "zh": ["相机使用希区柯克变焦", "希区柯克变焦"],
            "en": ["Dolly zoom shot", "Camera uses Hitchcock zoom"]
        },
        '快摇': {
            "zh": ["相机快速摇动捕捉场景", "画面进行快摇运镜", "快摇"],
            "en": ["The camera whip pans rapidly", "Fast whip pan shot"]
        },
        '悠悠球式变焦': {
            "zh": ["相机悠悠球式变焦", "悠悠球式变焦"],
            "en": ["Yoyo-style zoom in and out"]
        },
        '摆动运镜': {
            "zh": ["相机做摆动式运镜", "镜头左右摆动拍摄", "摆动运镜"],
            "en": ["The camera swings side to side", "Swinging camera motion"]
        },
        '斯坦尼康': {
            "zh": ["斯坦尼康运镜效果", "斯坦尼康运镜"],
            "en": ["Steadicam shot"]
        },
        '旋转运镜': {
            "zh": ["相机进行旋转运镜", "旋转运镜"],
            "en": ["The camera rotates", "Arc shot"]
        },
        '焦点转换': {
            "zh": ["镜头焦点发生转换", "焦点转换"],
            "en": ["Rack focus", "Focus shift", "Focus transitions"]
        },
        '背景虚化': {
            "zh": ["背景被虚化处理", "背景虚化"],
            "en": ["The background is blurred", "Blurred background"]
        },

    }

    # 运镜类型常量
    TYPE_NORMAL = "camera_movement_classification"  # 普通运镜
    TYPE_SPECIAL = "special_camera_movement_classification"  # 特殊运镜（背景虚化、旋转运镜等）

    # 运镜阈值常量
    THRESHOLD_NORMAL = 0.8  # 普通运镜阈值
    THRESHOLD_SPECIAL = 0.95  # 特殊运镜阈值

    I2V_EXCLUDE_KEYS = ["背景虚化", "鸟瞰/头顶", "静止", "无人机视角", "微距", "过肩镜头", "高角度/俯拍", "低角度/仰拍",
                        "鱼眼镜头"]
    T2V_EXCLUDE_KEYS = ["背景虚化"]

    @staticmethod
    def get_camera_movement_desc(camera_movement, lang):
        other = "OTHER"
        if camera_movement == other:
            return None
        assert camera_movement in CameraMovement.class_to_desc, f"Unsupported camera movement: {camera_movement}"
        if lang == "zh":
            candidate = CameraMovement.class_to_desc[camera_movement]['zh']
        elif lang == "en":
            candidate = CameraMovement.class_to_desc[camera_movement]['en']
        else:
            raise NotImplementedError(f"Unsupported language: {lang}")
        return random.choice(candidate)


class CaptionAug:

    def __init__(self,
                 caption_sample_ratio=None,
                 ocr_only_long_caption=False,
                 num_replace_rate=0,
                 random_caption_tag_order=False,
                 logger=None,
                 ):
        """
        Args:
            caption_sample_ratio : 结构化caption采样比例
        """
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.caption_sample_ratio = caption_sample_ratio
        if isinstance(caption_sample_ratio, str):
            self.caption_sample_ratio = json.loads(caption_sample_ratio)

        # Predefined keys
        self.caption_keys = {"short_caption", "medium_caption", "long_caption", "long_long_caption"}
        self.tag_keys = {'speed_mode', 'shot_type', 'shot_angle', 'viewpoint', 'camera_movement', 'composition', 'light', 'style', 'color_palette', 'atmosphere', 'background', 'ip'}
        self.predefined_keys = self.caption_keys.union(self.tag_keys)

        # User-defined caption sample ratio(csr) keys
        csr_keys = set([key for key, value in self.caption_sample_ratio.items() if value > 0])
        if csr_keys - self.predefined_keys:
            raise NotImplementedError(f"Unexpected keys in caption_sample_ratio: {csr_keys - self.predefined_keys}")
        self.valid_caption_keys = csr_keys.intersection(self.caption_keys)
        self.valid_tag_keys = csr_keys.intersection(self.tag_keys)

        self.logger = logger
        self.logger.info(
            "CaptionAug using caption sample ratio: {}".format(json.dumps(self.caption_sample_ratio))
        )
        self.ocr_only_long_caption = ocr_only_long_caption
        self.num_replace_rate = num_replace_rate
        self.random_caption_tag_order = random_caption_tag_order

    @staticmethod
    def safe_load_string(text):
        """加载结构化caption数据, json or xml格式"""
        if isinstance(text, dict):
            return text
        try:
            text = json.loads(text)
        except Exception as e:
            raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        if type(text) is str:
            try:
                text = eval(text)
                if type(text) is str:
                    text = text.replace("'", "\"")
                    text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got eval str: {}".format(str(e), text))

        if not isinstance(text, dict):
            print('load caption error', text)

        if "caption" in text:
            if isinstance(text["caption"], str):
                text = json.loads(text["caption"])
            else:
                text = text["caption"]

        # reload if still string
        if isinstance(text, str):
            try:
                text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        return text

    def random_replace_num(self, long_short_caption):
        """是否随机替换数字"""
        if self.num_replace_rate == 0 or self.ocr_in_caption(long_short_caption):
            return long_short_caption

        num_tag_map = {
            # "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",

            # "二": "2",
            # "三": "3",
            # "四": "4",
            # "五": "5",
            # "六": "6",
            # "七": "7",
            # "八": "8",
            # "九": "9",
            # "十": "10",
        }

        # 使用正则表达式匹配单词边界
        def replace_num(match):
            word = match.group(0).lower()
            if word in num_tag_map:
                # 对每个匹配到的数字按概率决定是否替换
                if random.random() < self.num_replace_rate:
                    return num_tag_map[word]
            return match.group(0)

        # 使用正则表达式替换，\b表示单词边界，这里中文匹配不生效
        pattern = r'\b(' + '|'.join(num_tag_map.keys()) + r')\b'
        new_caption = re.sub(pattern, replace_num, long_short_caption, flags=re.IGNORECASE)
        return new_caption

    @staticmethod
    def strip_zh(text):
        text = text.replace("。，", "，").replace("，。", "。").replace("。。", "。").replace("，，", "，")
        text = re.sub(r'\s+,', ' ', text).strip()
        text = text.lstrip("，。 ")
        return text

    @staticmethod
    def strip_en(text):
        text = text.replace(".,", ",").replace(",.", ".").replace("..", ".").replace(",,", ",")
        text = re.sub(r'\s+,', ' ', text).strip()
        text = text.lstrip(",. ")
        return text

    def ocr_in_caption(self, caption):

        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, caption)
        result = [match[0] or match[1] for match in matches]

        if result:
            return True
        return False

    @staticmethod
    def has_repeat(text, chunk_size=20, min_repeats=10):
        """
        检测文本中是否存在某个长度为 chunk_size 的模式等间隔连续重复出现至少 min_repeats 次。

        :param text: 输入字符串。
        :param chunk_size: 要查找的重复块的固定长度。
        :param min_repeats: 最小连续重复次数。
        :return: 是否存在满足条件的连续重复模式。

        算法思路：
            1. 记录每个模式出现的所有位置
            2. 检查是否有连续 min_repeats 个位置的差值相同（等间隔出现）
            3. 如果存在重复，则返回True，否则返回False

        """
        n = len(text)

        if n < chunk_size:
            return False

        # 记录每个模式出现的位置
        positions = {}
        for i in range(n - chunk_size + 1):
            pattern = text[i:i + chunk_size]
            if pattern not in positions:
                positions[pattern] = []
            positions[pattern].append(i)

        # 检查每个模式
        for pos_list in positions.values():
            if len(pos_list) < min_repeats:
                continue

            # 检查是否有连续 min_repeats 个位置的差值相同
            for i in range(len(pos_list) - min_repeats + 1):
                # 计算第一个差值
                diff = pos_list[i + 1] - pos_list[i]

                # 检查后续差值是否都相同
                is_uniform = True
                for j in range(i + 2, i + min_repeats):
                    if pos_list[j] - pos_list[j - 1] != diff:
                        is_uniform = False
                        break

                if is_uniform:
                    return True

        return False

    def replace_en_style(self, en_caption_dict, zh_caption_dict):
        try:
            style_en = en_caption_dict['style'].strip()
            style_zh = zh_caption_dict['style'].strip()
            if 'photo' in style_en.lower() and '摄影' not in style_zh:
                en_caption_dict['style'] = ''
        except Exception as e:
            self.logger.info(f'encounterd parsing error:{e}')
        return en_caption_dict

    def is_collage_only_longlong_caption(self, caption_key_candidates, caption_dict, lang):
        output_keys = None
        full_caption = ' '.join([caption_dict[key] for key in caption_key_candidates])

        # 拼接图, 只用 long/long_long_caption
        if (
                lang == "en" and ("collage" in full_caption or "triptych" in full_caption or "composite" in full_caption)
        ) or (
                lang == "zh" and ("拼贴" in full_caption or "三联画" in full_caption or "复合" in full_caption or "四格" in full_caption or "拼接" in full_caption or "并排" in full_caption)
        ):
            output_keys = [key for key in caption_key_candidates if key in ["long_caption", "long_long_caption"]]
        if output_keys is not None:
            return output_keys
        else:
            return caption_key_candidates

    def has_ocr_only_longlong_caption(self, caption_dict, caption_key_candidates):
        output_keys = None
        # 如果ocr_only_long_caption为True，则有ocr文字只使用long_caption和long_long_caption
        has_long_caption = 'long_caption' in caption_dict or 'long_long_caption' in caption_dict
        long_caption = caption_dict.get('long_caption','')
        longlong_caption = caption_dict.get('long_long_caption','')
        if has_long_caption and (self.ocr_in_caption(long_caption) or self.ocr_in_caption(longlong_caption)):
            output_keys = [key for key in caption_key_candidates if key in ["long_caption", "long_long_caption"]]
        if output_keys is not None:
            return output_keys
        else:
            return caption_key_candidates

    def random_compose(self, caption_dict, lang, return_key=False, caption_keys=None, tag_keys=None,
                       ignore_tag_prob=False, caption_keys_prob=None):
        assert lang in {"zh", "en"}, f"Unsupported language: {lang}"
        if caption_keys is not None:
            assert isinstance(caption_keys, list), f"caption_keys must be a list: {caption_keys}, got {type(caption_keys)}"
            valid_caption_keys = caption_keys
        else:
            valid_caption_keys = self.valid_caption_keys
        if tag_keys is not None:
            assert isinstance(tag_keys, list), f"tag_keys must be a list: {tag_keys}, got {type(tag_keys)}"
            valid_tag_keys = tag_keys
        else:
            valid_tag_keys = self.valid_tag_keys

        caption_key_candidates = []
        tag_candidates = {}
        for key in caption_dict:
            if key in valid_caption_keys:
                if caption_dict[key] and caption_dict[key] != "" and caption_dict[key].lower() != "none" and caption_dict[key] != "无":
                    caption_key_candidates.append(key)
            elif key in valid_tag_keys:
                if random.random() < self.caption_sample_ratio[key]:
                    tag = caption_dict[key].strip()
                    if tag != "" and tag.lower() != "none" and tag != "无":
                        tag_candidates[key] = tag.strip()

        if self.ocr_only_long_caption:
            caption_key_candidates = self.has_ocr_only_longlong_caption(caption_dict, caption_key_candidates)
        caption_key_candidates = self.is_collage_only_longlong_caption(caption_key_candidates, caption_dict, lang)

        if not caption_key_candidates or len(caption_key_candidates) == 0:
            raise ValueError(f"caption_key_candidates is empty! valid_caption_keys: {caption_keys}, caption_key_candidates: {caption_key_candidates}")

        # from all_captions, select by sample ratio weights
        selected_key = random.choices(
            caption_key_candidates,
            weights=[self.caption_sample_ratio[key] for key in caption_key_candidates],
        )[0]
        caption = caption_dict[selected_key].strip()
        if self.has_repeat(caption):
            raise ValueError(f"caption has repeat: {caption}")

        if len(tag_candidates) > 0:
            tag_candidates_lst = list(tag_candidates.values())
            random.shuffle(tag_candidates_lst)
        else:
            tag_candidates_lst = []

        if self.random_caption_tag_order:
            final_string = [SEPARATORS[lang]['comma'].join(tag_candidates), caption]
            random.shuffle(final_string)
            if lang == "zh":
                caption = self.strip_zh(SEPARATORS[lang]['period'].join(final_string))
            else:
                caption = self.strip_en(SEPARATORS[lang]['period'].join(final_string))
        else:
            if lang == "zh":
                caption = self.strip_zh(''.join([caption] + tag_candidates_lst))
            else:
                caption = self.strip_en(' '.join([caption] + tag_candidates_lst))

        # Remove meaningless characters
        caption = caption.strip()
        caption = caption.replace("\\N", "").strip("，,")

        if return_key:
            return CaptionOut(
                caption=caption, lang=lang, key=selected_key, tag_keys=list(tag_candidates.keys()),
            )
        return caption

    def caption_zh_process(self, caption_dict):
        # https://iwiki.woa.com/p/4015648283
        speed_mode_filter = lambda x: x in ['正常', '动态']
        shot_angle_filter = lambda x: '平视' in x
        viewpoint_filter = lambda x: '第三人称' in x
        should_remove_filters = {
            'speed_mode': speed_mode_filter,
            'shot_angle': shot_angle_filter,
            'viewpoint': viewpoint_filter,
        }
        remove_keys = []
        for key in caption_dict.keys():
            if key in should_remove_filters:
                func = should_remove_filters[key]
                if func(caption_dict[key]):
                    remove_keys.append(key)
        for key in remove_keys:
            caption_dict.pop(key)
        # 中文忽略camera_movement维度
        if "camera_movement" in caption_dict:
            caption_dict.pop("camera_movement")
        return caption_dict

    def _extract_prediction_from_classification(self, raw_string, classification_key, threshold):
        """
        从新格式的分类结果中提取预测结果，基于 top3_labels_and_probs 和阈值判断

        新格式示例:
        {
            "code": 200,
            "message": "success",
            "request_id": "xxx",
            "data": [{
                "<classification_key>": {
                    "code": 200,
                    "value": {
                        "prediction": "摇_右",
                        "prediction_use_threshold": "OTHER",
                        "top3_labels_and_probs": [
                            {"背景虚化": 0.9982503056526184},
                            {"旋转运镜": 0.00220846151933074},
                            {"摆动运镜": 0.0018017652910202742}
                        ],
                        ...
                    },
                    "version": "xxx"
                }
            }]
        }

        Args:
            raw_string: JSON 字符串
            classification_key: 分类结果的 key（如 "camera_movement_classification"）
            threshold: 置信度阈值，top1 概率 >= threshold 时返回该 label

        Returns:
            预测的 label，如果不满足阈值则返回 None
        """
        try:
            data = self.safe_load_string(raw_string)
            value_dict = None

            # 新格式: data[0].<classification_key>.value
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                classification_data = data["data"][0].get(classification_key, {})
                if isinstance(classification_data, dict) and "value" in classification_data:
                    value_dict = classification_data["value"]

            # 旧格式: 直接从顶层获取
            if value_dict is None:
                value_dict = data

            # 从 top3_labels_and_probs 取 top1
            top3 = value_dict.get("top3_labels_and_probs")
            if top3 and isinstance(top3, list) and len(top3) > 0:
                # top3[0] 格式: {"label_name": prob}
                top1_item = top3[0]
                if isinstance(top1_item, dict) and len(top1_item) > 0:
                    label, prob = next(iter(top1_item.items()))
                    if prob >= threshold:
                        return label
                    else:
                        return None

            # 如果没有 top3_labels_and_probs，回退到 prediction_use_threshold
            prediction = value_dict.get('prediction_use_threshold')
            if prediction == "OTHER":
                return None
            return prediction

        except Exception as e:
            self.logger.warning(f'encountered parsing {classification_key} error: {e}')
            return None

    def process_camera_movement(self, raw_string, lang, movement_type, caption_type="t2v"):
        """
        处理运镜分类字段

        Args:
            raw_string: JSON 字符串
            lang: 语言 ("en" 或 "zh")
            movement_type: 运镜类型，使用 CameraMovement 中的常量:
                - CameraMovement.TYPE_NORMAL: 普通运镜 (阈值 0.8)
                  旧格式: '{"prediction": "平移_左", "prediction_use_threshold": "OTHER", ...}'
                  新格式: '{"code": 200, "data": [{"camera_movement_classification": {"value": {...}}}]}'
                - CameraMovement.TYPE_SPECIAL: 特殊运镜 (阈值 0.95)
                  格式: '{"code": 200, "data": [{"special_camera_movement_classification": {"value": {...}}}]}'

        Returns:
            运镜描述字符串，如果不满足阈值则返回 None
        """
        # 根据类型获取阈值
        if movement_type == CameraMovement.TYPE_NORMAL:
            threshold = CameraMovement.THRESHOLD_NORMAL
        elif movement_type == CameraMovement.TYPE_SPECIAL:
            threshold = CameraMovement.THRESHOLD_SPECIAL
        else:
            self.logger.warning(f'Unknown movement_type: {movement_type}')
            return None

        prediction = self._extract_prediction_from_classification(
            raw_string,
            movement_type,
            threshold
        )
        if prediction is None:
            return None

        if caption_type == "t2v":
            if prediction in CameraMovement.T2V_EXCLUDE_KEYS:
                return None

        return CameraMovement.get_camera_movement_desc(prediction, lang)

    def update_camera_movement(self, caption_dict, lang):
        # 收集所有运镜描述 (t2v)
        camera_movement_descs = {"t2v": []}

        # 运镜字段与类型的映射
        CAMERA_MOVEMENT_FIELDS = {
            'camera_movement': CameraMovement.TYPE_NORMAL,  # 普通运镜
            'camera_movement_special': CameraMovement.TYPE_SPECIAL  # 特殊运镜
        }

        # 处理所有运镜字段
        external_camera_movement_dict = caption_dict['_inject_camera_']
        for field_name, movement_type in CAMERA_MOVEMENT_FIELDS.items():
            if field_name in external_camera_movement_dict:
                raw_string = external_camera_movement_dict.get(field_name)
                cm_t2v = self.process_camera_movement(raw_string, lang, movement_type, caption_type="t2v")
                if cm_t2v:
                    camera_movement_descs["t2v"].append(cm_t2v)

        # 合并运镜描述
        if camera_movement_descs["t2v"]:
            caption_dict['camera_movement'] = SEPARATORS[lang]['comma'].join(camera_movement_descs["t2v"])

        return caption_dict

    def caption_aug(self, raw_string: str | dict, lang, **kwargs):
        caption_dict = self.safe_load_string(raw_string)
        if lang == "zh":
            caption_dict = self.caption_zh_process(caption_dict)

        if '_inject_camera_' in caption_dict:
            caption_dict = self.update_camera_movement(caption_dict, lang)

        if kwargs.get("return_key"):
            if (pattern := kwargs.get("pattern")) is not None:
                raise NotImplementedError()
            else:
                caption = self.random_compose(caption_dict, lang, return_key=True)
        else:
            caption = self.random_compose(caption_dict, lang)

        caption = self.random_replace_num(caption)
        return caption
