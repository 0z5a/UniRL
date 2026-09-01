# =======================================================
# Caption process script for raw_text with inject_caption pattern.
#
# Data format:
#   {
#     "raw_text": "...some text... /{inject_caption}filename.pdf ...more text...",
#     "raw_text_lang": "en",
#     "format": "single-column" | "double-column" | "",   # optional
#     "caption_result": {
#       "filename.pdf": {
#         "zh": {"long_long_caption": "...", "long_caption": "...", ...},
#         "en": {"long_long_caption": "...", "long_caption": "...", ...}
#       }
#     }
#   }
#
# Processing pipeline:
#   1. The /{inject_caption}XXX patterns in raw_text are replaced with sampled
#      captions from caption_result[XXX][lang][caption_type], where lang and
#      caption_type are selected by configurable probabilities.
#      caption_result has multiple keys (e.g. "App_plot2.png", "isco_alpha13.pdf");
#      each key corresponds to one pattern: /{inject_caption}<key>. We iterate
#      over caption_result keys and replace each pattern separately.
#   2. If the "format" field is present and non-empty, a localized descriptive
#      prefix is randomly sampled from FORMAT_TEMPLATES and prepended to the
#      caption text. Each (lang, format) pair has multiple paraphrased templates
#      to increase diversity during training.
# =======================================================

import json
import random
import re

try:
    from .caption_base import CaptionOut
except ImportError:
    from caption_base import CaptionOut

INJECT_TAG = "/{inject_caption}"

FORMAT_TEMPLATES = {
    "en": {
        "single-column": [
            "Generate an academic paper page with a single-column layout.",
            "Create a single-column academic paper page.",
            "Produce a scholarly paper page in single-column format.",
            "Render an academic document page using a single-column layout.",
            "Generate a research paper page arranged in a single column.",
            "Create a single-column formatted scholarly document page.",
            "Compose an academic page with text arranged in one column.",
            "Generate a paper page featuring a single-column typesetting style.",
            "Design an academic paper page laid out in a single column.",
            "Output a research paper page that uses single-column formatting.",
            "Synthesize a one-column academic paper page.",
            "Construct an academic document page with a single-column structure.",
            "Generate a scholarly article page typeset in one column.",
            "Produce a single-column research document page.",
            "Create an academic manuscript page with single-column alignment.",
            "Generate a journal paper page using a one-column design.",
            "Render a single-column scholarly page layout.",
            "Build an academic paper page formatted as a single column.",
            "Generate a page of an academic paper in single-column style.",
            "Create a research paper layout with one column per page.",
        ],
        "double-column": [
            "Generate an academic paper page with a double-column layout.",
            "Create a double-column academic paper page.",
            "Produce a scholarly paper page in double-column format.",
            "Render an academic document page using a two-column layout.",
            "Generate a research paper page arranged in two columns.",
            "Create a double-column formatted scholarly document page.",
            "Compose an academic page with text arranged in two columns.",
            "Generate a paper page featuring a double-column typesetting style.",
            "Design an academic paper page laid out in two columns.",
            "Output a research paper page that uses double-column formatting.",
            "Synthesize a two-column academic paper page.",
            "Construct an academic document page with a double-column structure.",
            "Generate a scholarly article page typeset in two columns.",
            "Produce a double-column research document page.",
            "Create an academic manuscript page with two-column alignment.",
            "Generate a journal paper page using a two-column design.",
            "Render a double-column scholarly page layout.",
            "Build an academic paper page formatted as two columns.",
            "Generate a page of an academic paper in double-column style.",
            "Create a research paper layout with two columns per page.",
        ],
    },
    "zh": {
        "single-column": [
            "生成一页单栏排版的学术论文。",
            "创建一页单栏格式的学术论文页面。",
            "生成一份采用单栏布局的论文页面。",
            "请生成单栏排版的学术文档页面。",
            "生成一页以单栏形式排列的研究论文。",
            "创建一页单栏样式的学术论文排版。",
            "生成一份单栏格式的学术论文版面。",
            "请生成一页按照单栏方式排版的论文。",
            "构建一页单栏结构的学术论文。",
            "输出一页单栏排列的学术论文页面。",
            "生成一页单列式排版的研究论文。",
            "请创建一页使用单栏版式的学术文献。",
            "生成一页采用单栏对齐方式的论文。",
            "生成一份以单栏形式呈现的学术论文页。",
            "请生成一页单栏式的学术期刊页面。",
            "创建一页单栏排版的科研论文版面。",
            "生成一页按单栏格式编排的学术论文。",
            "请输出一页单栏设计的学术论文排版。",
            "生成一页具有单栏版面的学术论文。",
            "生成一份单栏排版风格的论文页面。",
        ],
        "double-column": [
            "生成一页双栏排版的学术论文。",
            "创建一页双栏格式的学术论文页面。",
            "生成一份采用双栏布局的论文页面。",
            "请生成双栏排版的学术文档页面。",
            "生成一页以双栏形式排列的研究论文。",
            "创建一页双栏样式的学术论文排版。",
            "生成一份双栏格式的学术论文版面。",
            "请生成一页按照双栏方式排版的论文。",
            "构建一页双栏结构的学术论文。",
            "输出一页双栏排列的学术论文页面。",
            "生成一页双列式排版的研究论文。",
            "请创建一页使用双栏版式的学术文献。",
            "生成一页采用双栏对齐方式的论文。",
            "生成一份以双栏形式呈现的学术论文页。",
            "请生成一页双栏式的学术期刊页面。",
            "创建一页双栏排版的科研论文版面。",
            "生成一页按双栏格式编排的学术论文。",
            "请输出一页双栏设计的学术论文排版。",
            "生成一页具有双栏版面的学术论文。",
            "生成一份双栏排版风格的论文页面。",
        ],
    },
}


class CaptionAug:
    def __init__(self, caption_sample_ratio=None, logger=None, inject_lang_ratio=None, wrap_triple_quotes=False):
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.caption_sample_ratio = caption_sample_ratio
        if isinstance(caption_sample_ratio, str):
            self.caption_sample_ratio = json.loads(caption_sample_ratio)

        self.inject_lang_ratio = inject_lang_ratio or {"en": 0.5, "zh": 0.5}
        if isinstance(self.inject_lang_ratio, str):
            self.inject_lang_ratio = json.loads(self.inject_lang_ratio)

        self.caption_keys = [k for k, v in self.caption_sample_ratio.items() if v > 0]
        self.caption_weights = [self.caption_sample_ratio[k] for k in self.caption_keys]

        self.lang_keys = list(self.inject_lang_ratio.keys())
        self.lang_weights = list(self.inject_lang_ratio.values())

        self.wrap_triple_quotes = wrap_triple_quotes

        self.logger.info(
            f"CaptionAug(rawtext) caption_sample_ratio={json.dumps(self.caption_sample_ratio)}, "
            f"inject_lang_ratio={json.dumps(self.inject_lang_ratio)}, "
            f"wrap_triple_quotes={self.wrap_triple_quotes}"
        )

    def _sample_inject_caption(self, caption_entry):
        """Sample a caption string from a single caption_result entry."""
        available_langs = [l for l in self.lang_keys if l in caption_entry]
        if not available_langs:
            return ""

        weights = [self.inject_lang_ratio[l] for l in available_langs]
        lang = random.choices(available_langs, weights=weights, k=1)[0]
        lang_data = caption_entry[lang]

        available_keys = [k for k in self.caption_keys if k in lang_data and lang_data[k]]
        if not available_keys:
            return ""

        weights = [self.caption_sample_ratio[k] for k in available_keys]
        selected_key = random.choices(available_keys, weights=weights, k=1)[0]

        return lang_data[selected_key]

    def _replace_inject_captions(self, raw_text, caption_result):
        """Replace /{inject_caption}<key> patterns in raw_text with sampled captions.

        Each key in caption_result corresponds to one pattern. We replace each
        pattern separately: for each key we sample once and replace every
        occurrence of /{inject_caption}<key> with that caption. Keys are
        processed by descending length so a longer key is not broken by
        replacing a shorter prefix first.
        """
        if not caption_result:
            return raw_text

        # Sort by key length descending to replace longer keys first (avoid "a" eating "a.png")
        keys_sorted = sorted(caption_result.keys(), key=len, reverse=True)
        result = raw_text
        for key in keys_sorted:
            replacement = self._sample_inject_caption(caption_result[key])
            if not replacement:
                continue
            # One pattern per key: literal /{inject_caption}<key>
            result = result.replace(INJECT_TAG + key, replacement)
        return result

    @staticmethod
    def _fix_json_escapes(s):
        """Fix unescaped backslashes (e.g. LaTeX \\in, \\mathbb) that are
        invalid JSON escape sequences."""
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    def caption_aug(self, raw_string, lang, **kwargs):
        if isinstance(raw_string, str):
            try:
                data = json.loads(raw_string, strict=False)
            except json.JSONDecodeError:
                data = json.loads(self._fix_json_escapes(raw_string), strict=False)
        else:
            data = raw_string

        raw_text = data.get("raw_text", "")
        if not raw_text:
            if kwargs.get("return_key"):
                return CaptionOut(caption=None, lang=lang)
            return None

        text_lang = data.get("raw_text_lang", lang)
        caption_result = data.get("caption_result", {})
        processed = self._replace_inject_captions(raw_text, caption_result)

        if self.wrap_triple_quotes:
            processed = f'"""{processed}"""'

        fmt = data.get("format", "")
        if fmt:
            templates = FORMAT_TEMPLATES.get(text_lang, FORMAT_TEMPLATES["en"])
            candidates = templates.get(fmt, [])
            if candidates:
                fmt_prefix = random.choice(candidates)
                processed = f"{fmt_prefix}\n{processed}"

        if kwargs.get("return_key"):
            return CaptionOut(caption=processed, lang=text_lang, key="raw_text")
        return processed


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    caption_sample_ratio = {
        "long_long_caption": 0.6,
        "long_caption": 0.2,
        "medium_caption": 0.1,
        "short_caption": 0.1,
    }

    aug = CaptionAug(caption_sample_ratio=caption_sample_ratio, logger=logger)

    # --- Test 1: normal JSON (properly escaped LaTeX) ---
    normal_input = json.dumps({
        "raw_text": "Solve $x^2 + 1 = 0$ for $x \\in \\mathbb{R}$.",
        "raw_text_lang": "en",
        "caption_result": {},
    })
    result = aug.caption_aug(normal_input, lang="en")
    print(f"[Test 1] Normal JSON:\n  {result}\n")

    # --- Test 2: broken JSON (unescaped LaTeX backslashes) ---
    broken_input = (
        '{"raw_text":"读句子联线。\\n1. $how$ $many$ $kites$ $can$ $you$ $see$?\\n2. $how$ $nice$/$beautiful$\\n3. $how$ $many$ $crayons$ $do$ $you$ $have$?\\n4. $open$ $it$ $and$ $see$\\n5. $look$ $at$ $my$ $new$ $crayons$	A 真漂亮\\nB 你能看见多少风筝\\nC 打开看看\\nD 看我的新蜡笔\\nE 你有多少只蜡笔","raw_text_lang":"zh","caption_result":{}}'
        # '{"raw_text":"绘制局部放大图时,应用 	圈出放大部分的部位。\\nA. 粗实线\\nB. 细实线\\nC. 细点画线\\nD. 波浪线","raw_text_lang":"zh","caption_result":{}}'
        # '{"raw_text":"I was grateful ( )	the stranger for returning my lost wallet.\\nA. for\\nB. to\\nC. with\\nD. of","raw_text_lang":"en","caption_result":{}}'
        # '{"raw_text":"Determine the largest $p \\in \\mathbb{R}$ such that the inequality \\[ x^4 + y^4 + z^4 + xyz(x + y + z) \\geq p(xy + yz + zx)^2 \\] holds for all $x, y, z \\in \\mathbb{R}$.","raw_text_lang":"en","caption_result":{}}'
    )
    print(f"[Test 2] Broken JSON input string:\n  {broken_input}")
    try:
        json.loads(broken_input)
        print("  json.loads succeeded directly (unexpected)")
    except json.JSONDecodeError as e:
        print(f"  json.loads fails as expected: {e}")
    result = aug.caption_aug(broken_input, lang="en")
    print(f"  caption_aug result:\n  {result}\n")

    # --- Test 3: inject_caption replacement with LaTeX backslashes in caption ---
    inject_input = json.dumps({
        "raw_text": "See Figure 1.\n/{inject_caption}fig1.pdf\nEnd.",
        "raw_text_lang": "en",
        "caption_result": {
            "fig1.pdf": {
                "en": {
                    "long_long_caption": "Plot of $P_{\\gamma \\to \\gamma}$ vs $\\displaystyle\\frac{E}{m_e}$.",
                    "long_caption": "A graph with \\geometry settings.",
                    "medium_caption": "Physics plot.",
                    "short_caption": "Plot.",
                }
            }
        },
    })
    result = aug.caption_aug(inject_input, lang="en")
    assert result is not None, "Test 3 failed: caption_aug returned None"
    assert "/{inject_caption}" not in result, f"Test 3 failed: inject tag not replaced in: {result}"
    print(f"[Test 3] Inject caption with LaTeX backslashes:\n  {result}\n")

    print("\nAll tests passed!")
