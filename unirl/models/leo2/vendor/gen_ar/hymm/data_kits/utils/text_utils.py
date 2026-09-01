import json
import re
from typing import TYPE_CHECKING, Union

from loguru import logger
if TYPE_CHECKING:
    from index_kits import ArrowIndexV2, MultiIndexV2
    from hymm.models.tokenizers import HunyuanMultimodalTokenizerFast

from .data_utils import DataMixin


class TextMixin(DataMixin):
    tokenizer: "HunyuanMultimodalTokenizerFast"
    task_kwargs: dict
    index_kwargs: dict
    index_manager: "Union[ArrowIndexV2, MultiIndexV2]"

    def setup_text(self, args):
        _ = args
        self.single_text_max_length = self.task_kwargs.get("single_text_max_length", 0)
        self.pre_extracted_tokens = self.task_kwargs.get(
            "pre_extracted_tokens", self.task_kwargs.get("pre_extract_tokens", False))
        # maximum token length for each input text
        self.input_max_token_length = self.task_kwargs["input_max_token_length"]
        # Whether to use special position tokens for bounding box.
        self.use_json_coord_tokens = self.task_kwargs.get("use_json_coord_tokens", True)

        # match patterns like `<quad>(123,456),(789,012)</quad>`
        self.quad_pattern = re.compile(
            r"<quad>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</quad>"
        )

    def preprocess_text(self, text, strip_think=False, strip_recaption=False, strip_image=False):
        if self.pre_extracted_tokens:
            assert isinstance(text, list), "Pre-extracted tokens should be a list of token ids."
            return text

        text = str(text)

        if strip_think:
            text = text.replace("<think>", "").replace("</think>", "")
        
        if strip_recaption:
            text = text.replace("<recaption>", "").replace("</recaption>", "")

        if strip_image:
            text = text.replace("<image>", "").replace("<img>", "")

        if self.single_text_max_length > 0:
            text = text[:self.single_text_max_length]

        return text.strip("\n ")
    
    def format_text_with_json(self, text, index)-> tuple[str, bool]:
        try:
            # 检查是否包含 <ref> 或 <quad> 标签
            has_ref = "<ref>" in text and "</ref>" in text
            has_quad = "<quad>" in text and "</quad>" in text
            
            if has_ref and has_quad:
                # 情况1和情况3：包含 <ref> 和 <quad>
                # 检查是否是纯标签格式（情况1）还是混合在文本中（情况3）
                # 使用正则表达式匹配 <ref>xxx</ref><quad>(x1,y1),(x2,y2)</quad> 模式，支持跨行匹配
                ref_quad_pattern = re.compile(
                    r"<ref>([^<]*)</ref>\s*<quad>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</quad>",
                    re.DOTALL
                )
                
                # 检查是否所有 <ref> 和 <quad> 都是成对出现的（情况1）
                all_matches = list(ref_quad_pattern.finditer(text))
                ref_count = len(re.findall(r"<ref>", text))
                quad_count = len(re.findall(r"<quad>", text))
                
                # 移除所有匹配的标签，检查是否还有其他文本内容
                text_without_tags = text
                for match in reversed(all_matches):
                    start, end = match.span()
                    text_without_tags = text_without_tags[:start] + text_without_tags[end:]
                # 移除所有剩余的标签（如果有未匹配的）
                text_without_tags = re.sub(r"<ref>.*?</ref>", "", text_without_tags)
                text_without_tags = re.sub(r"<quad>.*?</quad>", "", text_without_tags)
                text_without_tags = text_without_tags.strip()
                
                # 如果所有 ref 和 quad 都成对出现，且没有其他文本内容，则是情况1
                if len(all_matches) == ref_count == quad_count and len(all_matches) > 0 and not text_without_tags:
                    # 情况1：纯标签格式，合并相同标签
                    label_to_boxes = {}
                    for match in all_matches:
                        label = match.group(1).strip()
                        x1, y1, x2, y2 = map(int, match.groups()[1:])
                        box = [x1, y1, x2, y2]
                        if label not in label_to_boxes:
                            label_to_boxes[label] = []
                        label_to_boxes[label].append(box)
                    
                    # 构建 JSON 数组
                    result = []
                    for label, boxes in label_to_boxes.items():
                        result.append({"labels": label, "bboxes": boxes})
                    
                    formatted_text = json.dumps(result, ensure_ascii=False)
                    return formatted_text, True
                else:
                    # 情况3：混合在文本中
                    formatted_text = text
                    # 从后往前替换，避免位置偏移问题
                    for match in reversed(all_matches):
                        label = match.group(1).strip()
                        x1, y1, x2, y2 = map(int, match.groups()[1:])
                        box = [x1, y1, x2, y2]
                        json_obj = json.dumps({"labels": label, "bboxes": [box]}, ensure_ascii=False)
                        # 替换整个匹配部分
                        start, end = match.span()
                        # 保留标签文本和 JSON
                        formatted_text = formatted_text[:start] + label + " " + json_obj + formatted_text[end:]
                    
                    return formatted_text, True
                    
            elif has_quad and not has_ref:
                # 情况2：只有检测框
                quad_matches = list(self.quad_pattern.finditer(text))
                if len(quad_matches) > 0:
                    # 检查是否只有 quad 标签，没有其他文本内容
                    text_without_quads = text
                    for match in reversed(quad_matches):
                        start, end = match.span()
                        text_without_quads = text_without_quads[:start] + text_without_quads[end:]
                    # 移除所有剩余的 quad 标签（如果有未匹配的）
                    text_without_quads = re.sub(r"<quad>.*?</quad>", "", text_without_quads, flags=re.DOTALL)
                    text_without_quads = text_without_quads.strip()
                    
                    if not text_without_quads:
                        # 纯 quad 格式，合并所有框
                        boxes = []
                        for match in quad_matches:
                            x1, y1, x2, y2 = map(int, match.groups())
                            boxes.append([x1, y1, x2, y2])
                        
                        result = {"boxes": boxes}
                        formatted_text = json.dumps(result, ensure_ascii=False)
                        return formatted_text, True
        except Exception as e:
            logger.error(f"{e.__class__.__name__}: {e}. ({index=})")
            return text, False
        
        return text, False

    def format_ocr_data(self, ocr_str, index) -> tuple[str, bool]:
        formatted_str = ocr_str
        try:
            matched = []
            for match in self.quad_pattern.finditer(ocr_str):
                matched.append({
                    "start": match.start(),
                    "end": match.end(),
                    "coords": tuple(map(int, match.groups())),   # [x0, y0, x1, y1]
                })
            if len(matched) > 0:
                cat_str = ""
                last_pos = 0
                valid = True    # x and y in [0, 1000]
                for m in matched:
                    cat_str += ocr_str[last_pos:m["start"]]
                    x0, y0, x1, y1 = m["coords"]
                    valid = valid and (0 <= x0 < x1 <= 1000) and (0 <= y0 < y1 <= 1000)
                    cat_str += "".join([
                        self.tokenizer.quad_token,
                        self.tokenizer.x_token(x0),
                        self.tokenizer.y_token(y0),
                        self.tokenizer.x_token(x1),
                        self.tokenizer.y_token(y1),
                        self.tokenizer.end_of_quad_token,
                    ])
                    last_pos = m["end"]
                cat_str += ocr_str[last_pos:]
                formatted_str = cat_str
            else:
                valid = False
        except Exception as e:
            # raise e
            logger.error(f"{e.__class__.__name__}: {e}. ({index=})")
            valid = False

        formatted_str = (formatted_str
                         .replace("<ref>", self.tokenizer.ref_token)
                         .replace("</ref>", self.tokenizer.end_of_ref_token))

        return formatted_str, valid

    def preprocess_text_with_ocr(self, text, index):
        # Process ocr
        valid = True
        # 单纯的检测不存在<ref></ref>；只有<quad></quad>
        if ("<ref>" in text and "</ref>" in text) or ("<quad>" in text and "</quad>" in text):
            if not self.use_json_coord_tokens:
                text, valid = self.format_ocr_data(text, index)
            else:
                text, valid = self.format_text_with_json(text, index)
        else:
            # If the text is in XML format and the special coordinate tokens are not used, convert it to JSON format.
            #  if the text ends with "XML format. or "XML format", replace it with "JSON format. or "JSON format".
            if self.use_json_coord_tokens:
                if  (text.endswith("XML format.") or text.endswith("XML format")):
                    # 将结尾的"XML format."或"XML format"替换为"JSON format."或"JSON format"； 只替换结尾的即可
                    text = text.rstrip("XML format.").rstrip("XML format")
                    text = text + " JSON format."
                text = text.replace("Latex format", "JSON format")
               
        return self.preprocess_text(text, strip_image=True), valid

    @staticmethod
    def preprocess_multilingual_text(msg, lang):
        # Find a text following the order: lang -> text -> alt_lang
        if f"text_{lang}" in msg and msg[f"text_{lang}"]:
            text = msg[f"text_{lang}"]
        elif "text" in msg and msg["text"]:
            text = msg["text"]
        else:
            alt_lang = "zh" if lang == "en" else "en"
            text = msg.get(f"text_{alt_lang}", "")
        if text is None:
            text = ""
        return text

    def preprocess_multilingual_text_with_ocr(self, msg, lang, index):
        # Find a text following the order: lang -> text -> alt_lang
        text = self.preprocess_multilingual_text(msg, lang)
        return self.preprocess_text_with_ocr(text, index)

    # ==================
    #   Tool Calls
    # ==================

    @staticmethod
    def format_tool_descriptions(tools):
        tools_descriptions = []
        for tool in tools:
            # Here we manually construct the tool description dict to make sure the order of key-value pairs.
            assert tool["type"] == "function", f"Only function type tools are supported, but got {tool['type']}."
            func = tool["function"]

            if func["name"] == "generate":
                properties = func["parameters"]["properties"]
                properties_dict = {}
                for key in ["source_image_indices_list", "recaption", "image_size", "image_ratio"]:
                    properties_dict[key] = {
                        "description": properties[key]["description"],
                        "type": properties[key]["type"],
                    }
                description_dict = {
                    "type": "function",
                    "function": {
                        "name": "generate",
                        "description": func["description"],
                        "parameters": {
                            "type": func["parameters"]["type"],
                            "properties": properties_dict,
                            "required": func["parameters"]["required"]
                        }
                    }
                }
                tools_descriptions.append(json.dumps(description_dict, ensure_ascii=False))

        tools_descriptions_repr = '\n'.join(tools_descriptions)
        return f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{tools_descriptions_repr}
</tools>

For function call returns, you should first print <tool_calls>
For each function call, you should return object like:
<tool_call>{{function-name}}<tool_sep>
<arg_key>{{arg-key-1}}</arg_key>
<arg_value>{{arg-value-1}}</arg_value>
<arg_key>{{arg-key-2}}</arg_key>
<arg_value>{{arg-value-2}}</arg_value>
...
</tool_calls>
At the end of function call returns, you should print </tool_calls>"""

    def format_tool_calls(
            self,
            tool_calls: dict | list[dict],
            tool_responses: dict | list[dict],
            index: int,
            **kwargs,
    ) -> tuple[str, list[dict], dict] | tuple[dict, list[dict], dict]:
        """
        For 'generate' function, the tool_calls and will be like:
        <tool_calls>
            <tool_call>generate<tool_sep>
                <arg_key>source_image_indices_list</arg_key>
                <arg_value>[0]</arg_value>
                <arg_key>recaption</arg_key>
                <arg_value>New caption</arg_value>
                <arg_key>image_size</arg_key>
                <arg_value><img_size_1024></arg_value>
                <arg_key>image_ratio</arg_key>
                <arg_value><img_ratio_16></arg_value>
            </tool_call>
        </tool_calls>
        <tool_responses>
            <tool_response>
                {"status_code": 200, "image": "<image>"}
            </tool_response>
        </tool_responses>
        """

        if isinstance(tool_calls, list):
            response_messages = []
            tool_contents = []
            return_status = {}
            for tool_call, tool_response in zip(tool_calls, tool_responses):
                call_content, cur_response_messages, return_kwargs = self.format_tool_calls(
                    tool_call, tool_response, index, **kwargs
                )
                tool_contents.append(call_content)
                response_messages.extend(cur_response_messages)
                # Update some status in kwargs for the next tool call
                kwargs = {**kwargs, **return_kwargs}
                return_status.update(return_kwargs)
            call_message = dict(
                type="gen_text",
                text="<tool_calls>" + "".join(tool_contents) + "</tool_calls>",
            )
            return call_message, response_messages, return_status

        elif isinstance(tool_calls, dict):
            return_kwargs = {}
            if tool_calls["type"] == "function":
                function = tool_calls["function"]
                func_name = function["name"]
                arg_list = []
                if func_name == "generate":
                    arg = function["arguments"]
                    arg_list.append(f"<arg_key>source_image_indices_list</arg_key><arg_value>{arg['source_image_indices_list']}</arg_value>")
                    arg_list.append(f"<arg_key>recaption</arg_key><arg_value>{arg['recaption']}</arg_value>")
                    # image_size and image_ratio need to be determined by the response images
                    assert tool_responses is not None, "tool_responses is required for generate function."
                    assert "image_list" in kwargs and "image_idx" in kwargs and "img_success" in kwargs, \
                        "image_list and image_idx and img_success are required in kwargs for generate function."

                    image_list = kwargs["image_list"]
                    image_idx = kwargs["image_idx"]
                    img_success = kwargs["img_success"]
                    gen_images = kwargs.get("gen_images", [])

                    gen_image, cur_img_success = self.get_image_with_size(
                        src=dict(cache_image=image_list[image_idx]),
                        random_crop=False,
                        target_size_type="image",
                        return_type="vae",
                        real_index=index,
                        column="cache_image",
                    )
                    gen_images.append(gen_image)
                    size_token = self.tokenizer.size_token(gen_image.i.base_size)
                    ratio_token = self.tokenizer.ratio_token(gen_image.i.ratio_index)
                    arg_list.append(f"<arg_key>image_size</arg_key><arg_value>{size_token}</arg_value>")
                    arg_list.append(f"<arg_key>image_ratio</arg_key><arg_value>{ratio_token}</arg_value>")

                    response_messages = [
                        dict(type="tool_response_text", text="<tool_response>{\"status_code\": 200, \"image\": \""),
                        dict(type="tool_response_gen_image", metadata=gen_image.i.tool_call_meta_info),
                        dict(type="tool_response_text", text="\"}</tool_response>"),
                    ]
                    return_kwargs["image_idx"] = image_idx + 1
                    return_kwargs["img_success"] = img_success and cur_img_success
                    return_kwargs["gen_images"] = gen_images

                else:
                    raise NotImplementedError(f"Function '{func_name}' is not implemented in format_tool_calls.")

                call_content = f"<tool_call>{func_name}<tool_sep>" + "".join(arg_list) + "</tool_call>"
                return call_content, response_messages, return_kwargs

            else:
                raise NotImplementedError(f"Tool call type '{tool_calls['type']}' is not implemented.")

        else:
            raise TypeError(f"tool_calls must be a list or a dict: {type(tool_calls)}")


# ----------------------------------------------------------------------
# Module-level text helpers (no class state required)
# ----------------------------------------------------------------------
def has_repeat(text: str, chunk_size: int = 20, min_repeats: int = 10) -> bool:
    """检测文本中是否存在某个长度为 chunk_size 的模式等间隔连续重复出现至少 min_repeats 次。

    :param text: 输入字符串。
    :param chunk_size: 要查找的重复块的固定长度。
    :param min_repeats: 最小连续重复次数。
    :return: 是否存在满足条件的连续重复模式。

    算法：
        1. 记录每个长度为 chunk_size 的窗口在文本里出现的所有位置
        2. 对位置数 >= min_repeats 的窗口，检查是否有连续 min_repeats 个位置等间隔
    """
    n = len(text)
    if n < chunk_size:
        return False

    positions = {}
    for i in range(n - chunk_size + 1):
        pattern = text[i:i + chunk_size]
        positions.setdefault(pattern, []).append(i)

    for pos_list in positions.values():
        if len(pos_list) < min_repeats:
            continue
        for i in range(len(pos_list) - min_repeats + 1):
            diff = pos_list[i + 1] - pos_list[i]
            if all(pos_list[j] - pos_list[j - 1] == diff
                   for j in range(i + 2, i + min_repeats)):
                return True
    return False