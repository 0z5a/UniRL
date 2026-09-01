import os
import json
from typing import Union, List, Optional
from PIL import Image

from hymm.models.reward_models.utils.gemini_api import request_batch
from hymm.models.reward_models.face_similarity import FaceSimilarityRewardModel


SYSTEM_PROMPTS = {
    ### 1. OCR任务
    "OCR_CN_1": """
    现在你是一个文字-图像生成一致性判别助手，我将给你两个输入，分别是prompt和基于prompt生成的图片。
    该prompt要求生成的图片中包含一些In-image text，请你根据prompt提取出来要求的In-image text是什么。
    然后根据生成的图片，判断该图片中的In-image text是否符合prompt的要求的In-image text。

    主要标准是：
    1. 生成的图片中的In-image text与prompt要求的完全一致，不能有错别字或者写法错误、乱码等。（比如要求的文字是"国"但是生成的图片中"国"字少了一个点，英文字母z上面多了一个点等）
    2. 不能有遗漏的字或者字符
    3. 不能有额外多余的字或者字符
    4. 文字的位置与prompt描述中的位置需要匹配

    **重要：** 如果图片中的是乱码或者写法不正确的字甚至无法识别为特定文字，也算作错误

    如果完全正确，输出"正确"，否则输出"错误"，不需要给出原因
    """,

    ### 2. OCR任务-2
    "OCR_CN_2": """
    现在你是一个文字-图像生成一致性视觉助手，我将给一个prompt和基于prompt生成的一些图片。
    该prompt要求生成的图片中包含一些In-image text，请你根据prompt提取出来要求的In-image text是什么。
    然后根据生成的这些图片，评估这些图片中的In-image text是否符合prompt的要求的In-image text，并为每张图片图片进行打分，打分范围是0到10分，0分表示完全不符合，10分表示完全符合。

    打分标准如下：
    1. 不能有遗漏的字或者字符。
    2. 不能有额外多余的字或者字符。
    3. 不能有错别字或者写法错误、乱码等。（比如要求的文字是"国"但是生成的图片中"国"字少了一个点，英文字母z上面多了一个点等）
    4. 文字的位置与prompt描述中的位置需要匹配。

    返回格式：
    1. 对于不同图片，按输入顺序依次返回结果，不同图片的评估结果之间通过换行符分隔。
    2. 对于每张图片，首先返回一个分数，分数使用<>标记然，后是一个简短的评估理由，例如"Image 1: <5>, 文字基本符合要求Prompt要求的"美美与共"，但是遗漏了一个字\"美\"，同时有一个错别字\"共\""。
    3. 返回结果的行数与图片数量一致，除去分数和评估理由外，不要有其他内容。
    """,

    "COUNT_CN_NO_REASON": """
    现在你是一个物体数量-图像生成一致性视觉助手，我将给一个prompt和基于prompt生成的图片。
    请根据prompt提取出要求的物体种类及数量，然后判断图片中的物体数量是否与prompt要求一致。
    
    打分标准如下：
    1. 物体的种类、数量与prompt描述完全一致，不能多也不能少。
    2. 如果prompt中存在多个需要判断数量的物体，则需要判断图片中是否所有这些物体的数量生成都正确。
    
    返回格式：
    首先判断prompt中物体数量是否与图片中实际的物体数量是否一致，如果一致，输出"正确"，否则输出"错误"，不需要给出具体原因。
    """,

    ### 3. 物体计数任务
    "COUNT_CN_WITH_REASON": """
    现在你是一个物体数量-图像生成一致性视觉助手，我将给一个prompt和基于prompt生成的图片。
    请根据prompt提取出要求的物体种类及数量，然后判断图片中的物体数量是否与prompt要求一致，并给出具体原因。
    
    打分标准如下：
    1. 物体的种类、数量与prompt描述完全一致，不能多也不能少。
    2. 如果prompt中存在多个需要判断数量的物体，则需要判断图片中是否所有这些物体的数量生成都正确。
    3. prompt中如果要求生成文字，则需要判断图片中是否生成了文字，并且生成的文字是否与prompt要求一致。
    
    返回格式：
    首先判断prompt中物体数量是否与图片中实际的物体数量是否一致，以及如果要求生成文字，则需要判断图片中是否生成了文字，并且生成的文字是否与prompt要求一致。
    如果一致，输出"正确"，否则输出"错误"，并给出具体原因。
    """,

    ### 4. 图像编辑任务准确度
    "Editing": '''You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
    All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

    You will have to give your output in this way (Keep your reasoning concise and short.):
    {
    "score" : ...,
    "reasoning" : "..."
    }
    RULES:

    Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
    The objective is to evaluate how successfully the editing instruction has been executed in the second image.

    Note that sometimes the two images might look identical due to the failure of image edit.


    From scale 0 to 10: 
    A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
    A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
    Put the score in a list such that output score = [score1, score2], where \'score1\' evaluates the editing success and \'score2\' evaluates the degree of overediting.

    Editing instruction: <INSTRUCTION_PLACEHOLDER>''',


    ### 5. 图像自然度与图像伪影
    "Naturalness_Artifacts": '''You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
    All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

    You will have to give your output in this way (Keep your reasoning concise and short.):
    {
    "score" : [...],
    "reasoning" : "..."
    }
    RULES:

    The image is an AI-generated image.
    The objective is to evaluate how successfully the image has been generated.

    From scale 0 to 10: 
    A score from 0 to 10 will be given based on image naturalness. 
    (
        0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
        10 indicates that the image looks natural.
    )
    A second score from 0 to 10 will rate the image artifacts. 
    (
        0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
        10 indicates the image has no artifacts.
    )
    Put the score in a list such that output score = [naturalness, artifacts]''',


    ### 6. 编辑-主体替换类
    "Editing_Subject_Replacement": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“主体替换”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？即替换了指令要求的物体，且替换后的新主体是否与指令描述的身份/类别完全一致？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对主体替换)**
    - '融合自然度': 新主体的边缘、光照和阴影是否与周围环境无缝融合？
    - '透视与比例正确性': 新主体的透视和大小比例在场景中是否合理？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "融合自然度": <SCORE>, "透视与比例正确性": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 7. 编辑-主体属性修改类
    "Editing_Subject_Attribute_Modification": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“主体属性修改”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？即修改后的属性是否与指令描述完全一致？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对主体属性修改)**
    - '修改结果真实性': 修改后的属性看起来是否真实自然？
    - '主体身份保持度': 修改属性后，主体的核心身份特征是否得以保留？
    - '修改范围精确度': 修改是否仅限于目标区域，未污染到周边？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE> },
        "专项维度评估": { "修改结果真实性": <SCORE>, "主体身份保持度": <SCORE>, "修改范围精确度": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 8. 编辑-主体删除类
    "Editing_Subject_Deletion": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“主体删除”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？即指令要求删除的主体是否被完全移除，无任何残留？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对主体删除)**
    - '背景填充一致性': 被移除区域填充的背景，在纹理、结构和颜色上是否与周围环境无缝衔接？
    - '填充区域无伪影': 填充区域是否存在模糊、扭曲、重复纹理等不自然的伪影？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "背景填充一致性": <SCORE>, "填充区域无伪影": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 9. 编辑-主体添加类
    "Editing_Subject_Addition": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“主体添加”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？即是否按照指令在正确的位置添加了正确的主体（类别、数量）？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对主体添加)**
    - '融合自然度': 新主体的边缘、光照和阴影是否与周围环境无缝融合？
    - '透视与比例正确性': 新主体的透视和大小比例在场景中是否合理？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "融合自然度": <SCORE>, "透视与比例正确性": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 10. 编辑-动作修改类（比如video2frame数据）
    "Editing_Action_Modification": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“动作修改”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？即修改后的动作是否符合指令描述？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对动作修改)**
    - '物理逻辑合理性': 新的主体动作是否符合物理逻辑？
    - '环境交互一致性': 新动作是否与环境产生了合理的交互（如坐在椅子上）？
    - '主体身份保持度': 修改动作后，主体的身份特征是否未发生改变？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "物理逻辑合理性": <SCORE>, "环境交互一致性": <SCORE>, "主体身份保持度": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 11. 编辑-场景修改（比如天气、季节、背景等）
    # '技术质量': 仅评判修复后的图像是否存在明显的技术缺陷、美学问题、主体或背景失真等等？（排除指令要求或者指令本身伴随的问题）
    "Editing_Scene_Modification": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“场景修改”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？即新场景是否与指令要求一致？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对场景修改)**
    - '全局效果一致性': 场景变化是否一致地应用到了整个画面？
    - '前景主体融合度': 前景主体是否与新场景在光照、颜色等方面自然融合？
    - '前景主体保持度': 前景主体的结构和细节是否被完好保留？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "全局效果一致性": <SCORE>, "前景主体融合度": <SCORE>, "前景主体保持度": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 12. 编辑-风格修改
    "Editing_Style_Modification": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“风格修改”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？图像是否成功应用了指令中指定的艺术风格？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 是否保持主体及背景内容结构跟原图的一致？

    **第二部分：专项维度评估 (针对风格修改)**
    - '原图内容可辨识度': 应用新风格后，原图的主要内容和结构是否仍然清晰可辨？
    - '风格一致性': 整个画面的风格应用是否统一？
    - '风格美学契合度': 最终生成的图像是否具有所应用风格应有的美感？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "原图内容可辨识度": <SCORE>, "风格一致性": <SCORE>, "风格美学契合度": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 13. 编辑-文本编辑
    "Editing_Text_Modification": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“文本编辑”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？是否准确地执行了添加、删除或修改文本的指令？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对文本编辑)**
    - '文本内容正确性': 添加或修改后的文本内容是否与指令中的文字完全一致？
    - '文本风格一致性': 添加的文本在字体、颜色、透视上是否与图片环境协调？
    - '背景修复质量': 删除或修改文本后，原背景区域是否被修复得自然无痕？若不涉及，则输出1。

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "文本内容正确性": <SCORE>, "文本风格一致性": <SCORE>, "背景修复质量": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 14. 编辑-图像修复（inpainting、去水印、去噪等等）
    "Editing_Image_Restoration": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“图像修复”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？（比如图像中的噪点、水印、划痕等缺陷是否被按照指令要求有效去除或修复？）
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对图像修复)**
    - '细节保留度': 在修复缺陷的同时，图像原有的重要细节是否被清晰地保留？
    - '无二次伪影': 修复过程是否引入了新的不自然伪影？
    - '色彩与亮度真实性': 修复后的图像色彩和亮度是否自然、真实？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "细节保留度": <SCORE>, "无二次伪影": <SCORE>, "色彩与亮度真实性": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 15. 编辑-光照修改
    "Editing_Lighting_Modification": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“光照修改”编辑的效果。您将收到[原始图片](即第一张图片)、[编辑后图片](即第二张图片)和[编辑指令]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”：
    **第一部分：通用维度评估**
    - '指令遵循度': 是否基本完成了指令的核心要求？新的光照效果是否符合指令描述（如方向、色温）？
    - '技术质量': 编辑后图像是否存在明显的技术缺陷？（比如模糊与像素化、伪影与失真、边缘接缝失真、颜色问题等等，除非是本身指令要求或者指令本身伴随）
    - '非编辑区稳定性': 主体及背景的非编辑区域是否保持与原图完全一致？

    **第二部分：专项维度评估 (针对光照修改)**
    - '光照效果真实性': 新的光照看起来是否真实，符合光学规律？
    - '光影逻辑一致性': 图像中所有物体的光影是否与新的光源保持逻辑一致？
    - '细节保持度': 在强光或阴影区域，图像的细节是否仍然可见，没有过曝或死黑？

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
        "专项维度评估": { "光照效果真实性": <SCORE>, "光影逻辑一致性": <SCORE>, "细节保持度": <SCORE> }
    },
    "reasoning": ...,
    }

    给定的[编辑指令]为：<INSTRUCTION_PLACEHOLDER>
    ''',


    ### 16. 编辑指令归类
    "Editing_Instruction_Classification": '''
    # 角色
    你是一个专业的图像编辑指令分类专家。

    # 任务
    你的任务是仔细分析用户输入的编辑指令，并将其准确地归类到以下10个预定义类别中的一个。你必须且只能从这10个类别中选择一个作为输出。

    # 10个预定义类别及定义
    1.  **主体替换**: 将图片中的主要对象A替换为另一个对象B。

    2.  **属性修改**: 对图片中某个主体的内在属性（如颜色、材质、大小、形状等）进行修改，但不改变其本质。
        - 例子: "把她的头发染成红色", "让这辆车变大一点", "给这个杯子换成玻璃材质"

    3.  **主体删除**: 从图片中移除一个或多个主要对象。

    4.  **主体添加**: 在图片中添加一个全新的、原本不存在的主要对象。

    5.  **动作修改**: 改变人物或动物的姿势、动作或表情。
        - 例子: "让她笑起来", "让他从坐着变成站着", "把他的手改成挥手的姿势"

    6.  **场景修改**: 对图片的背景或整体环境进行修改或替换。
        - 例子: "把背景换成海滩", "让房间里变得更整洁", "在地上铺上地毯"

    7.  **风格修改**: 改变图片的整体艺术风格、色调或氛围。

    8.  **文本编辑**: 在图片上添加、删除或修改文字。

    9.  **图像修复**: 修复图片中的瑕疵、破损或低质量问题。
        - 例子: "把照片上的划痕去掉", "让这张老照片变清晰", "去掉水印", "补全mask区域"

    10. **光照修改**: 调整图片的整体或局部光照效果。
        - 例子: "让光线从左边打过来", "增加黄昏的感觉", "提亮人脸"

    # 输出要求
    - **严格**按照上面给出的10个类别名称进行输出。
    - **不要**添加任何额外的解释、编号或文字。
    - 直接输出最匹配的类别名称。

    # 用户指令
    【<INSTRUCTION_PLACEHOLDER>】

    # 分类结果
    ''',

    ### 17. t2i prompt语义拆分
    "T2I_Prompt_Semantic_Decomposition": '''
    # 角色
    你是一个精通自然语言理解（NLU）的AI专家，专门负责将用于图像生成的文本prompt进行深度的语义结构化拆解。

    # 任务
    你的任务是接收一个文本prompt，并将其严格按照下面定义的语义概念进行拆解，保证**每个拆解的元素是一个能独立具有明确含义的词组或短句**，你必须将拆解结果以一个结构化的JSON对象输出。

    # 语义概念定义
    你需要在prompt中识别以下概念：

    - **`主要主体`**: 图像中最核心、最引人注目的一个或多个主体。
        - `名词`: 主体的名称。
        - `关键属性`: 定义主体核心身份或特征的描述，需要加上修饰的主体的名称。
        - `其他属性`: 对主体的次要补充描述，需要加上修饰的主体的名称。
        - `动作`: 主体正在进行的动作，需要加上修饰的主体的名称。

    - **`次要主体`**: 图像中起陪衬、互动或背景作用的一个或多个次要主体。
        - `名词`: 次要主体的名称。
        - `属性`: 对次要主体的描述，需要加上修饰的次要主体的名称。
        - `动作`: 次要主体正在进行的动作，需要加上修饰的次要主体的名称。

    - **`场景`**: 描述图像发生的整体环境或背景。
        - `名词`: 场景的名称。
        - `属性`: 对场景的补充描述，需要加上修饰的场景的名称。

    - **`镜头`**: 描述拍摄视角、镜头类型或距离的词语。

    - **`风格`**: 描述图像整体艺术风格、流派或质感的词语。

    # 核心规则
    1.  **识别并归类**: 根据上述概念对prompt进行总结，并将总结后的结果以JSON格式输出。
    2.  **每个概念支持多个结果**: 必须是**数组（Array）**的形式输出。
    3.  **按需输出**: 如果在prompt中**完全没有找到**对应某个概念的信息，输出空数组 `[]`。
    4.  (!非常重要) **拆解的元素的语义完整性**: 必须保证拆解的元素是独立的，不能有任何的歧义的词组或短句。比如prompt是'一只黑色的狗'，关键属性应该是['一只狗', '黑色的狗']，而非['一只', '黑色']

    # 示例
    **待分析的Prompt:**
    `一位少女在一片花海里，依偎着一株大树，她的左边还有一株类似的树，画面以3D风格展现。`

    **理想的JSON输出:**
    ```json
    {
        "main_subjects": [
            {
            "noun": ["少女", "大树"],
            "key_attributes": ['一位少女'],
            "other_attributes": [
                "少女在花海里",
                "被少女依偎着的一颗大树",
                "少女依偎着的大树左边还有一颗类似的大树",
            ],
            "action": ["少女依偎着大树"]
            }
        ],
        "secondary_subjects": [
            {
            "noun": [],
            "attributes": [],
            "action": [],
            }
        ],
        "scene": {
            "noun": ["花海"],
            "attributes": ["一片花海"]
        },
        "camera": [],
        "style": ["3D风格"],
    }

    # 文生图prompt
    【<INSTRUCTION_PLACEHOLDER>】

    # 输出结果
    ''',

    ### 18. subject driven图片一致性
    "Subject_Driven_Consistency": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“Subject Driven”的图像生成效果。您将收到[文本prompt]、[Subject原始图像](即第一张图片)和[生成的图像](即第二张图片)。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”。注意：为了使图像看起来自然，可以允许Subject有视角、大小上的改变。
    **通用维度评估**
    - 'Subject形状一致性': 生成的图像中的subject是否与原始图像中的subject在形状上保持一致？（即没有明显的形状扭曲或变形）
    - 'Subject外观一致性': 生成的图像中的subject是否与原始图像中的subject在外观上保持一致？（即没有颜色、纹理、质感上的不一致）
    - 'Subject细节一致性': 生成的图像中的subject是否与原始图像中的subject在细节上保持一致？（即没有明显的细节缺失或过度添加）
    - '自然度': 生成的图像中的subject是否与背景自然融合？（即没有明显的拼接痕迹或不自然的边缘，subject的出现不会突兀）
    - '物理规律': 生成的图像中的subject是否遵循物理规律？（即大小、透视等方面与背景相比符合物理规律，除非是prompt要求修改）

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "通用维度评估": { "Subject形状一致性": <SCORE>, "Subject外观一致性": <SCORE>, "Subject细节一致性": <SCORE>, "自然度": <SCORE>, "物理规律": <SCORE>}
    },
    "reasoning": ...,
    }

    给定的[文本prompt]为：<INSTRUCTION_PLACEHOLDER>
    ''',

    ### 19. subject driven任务语义对齐度
    "Semantic_Alignment": '''
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“图像生成”的语义对齐效果。您将收到[生成的图像]和[Prompt语义拆分内容]。

    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”。注意：如果某个[Prompt语义拆分内容]的概念为空，可以直接输出1。
    **专项维度评估 (针对每个语义概念)**
    - 'main_subjects对齐度': 生成的图像与"main_subjects"的语义概念是否对齐？（即图像中是否包含了所有的主要主体，并且它们的属性、动作等符合语义概念）
    - 'secondary_subjects对齐度': 生成的图像与"secondary_subjects"的语义概念是否对齐？（即图像中是否包含了所有的次要主体，并且它们的属性、动作等符合语义概念）
    - 'scene对齐度': 生成的图像与"scene"的语义概念是否对齐？（即图像中的场景是否符合语义概念）
    - 'camera对齐度': 生成的图像与"camera"的语义概念是否对齐？（即图像中的镜头视角、类型等是否符合语义概念）
    - 'style对齐度': 生成的图像与"style"的语义概念是否对齐？（即图像的艺术风格是否符合语义概念）

    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
    "score": {
        "专项维度评估": { "main_subjects对齐度": <SCORE>, "secondary_subjects对齐度": <SCORE>, "scene对齐度": <SCORE>, "camera对齐度": <SCORE>, "style对齐度": <SCORE> }
    },
    "reasoning": ...,
    }
    ''',

    ### 20. OCR任务
    "OCR_3": """
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次”图像生成“的语义对齐效果。您将收到[生成的图像]和[文本Prompt]。[文本Prompt]会要求[生成的图像]中包含某些特定的文字内容，可能是中文也可能是英文，您需要特别注意。
    请根据以下问题进行评估，1代表“是/成功/好”，0代表“否/失败/差”。
   
    **专项维度评估**
    - '语义一致性': 除了文字内容外，[生成的图像]是否与[文本Prompt]的其他描述保持一致？（即图像主体、场景、风格等与[文本Prompt]基本保持一致）
    - '文字数量正确性': [生成的图像]中包含的文字数量是否与[文本Prompt]中要求的文字内容的数量一致？（即没有遗漏或多余的文字，除非是伴随生成了一些较小字体为了增加图像美观度）
    - '文字内容正确性': [生成的图像]中包含的文字在内容上是否与[文本Prompt]中的要求一致？（即需要逐字对比，确保没有任何拼写错误）
    - '文字可读性': [生成的图像]中包含的文字在形状和结构上是否正确、清晰可读？（即没有明显的形状扭曲、变形、重叠、模糊或其他影响可读性的因素，除非是[文本Prompt]要求特定风格、字体等）
    - '文字位置正确性': [生成的图像]中包含的文字位置是否与[文本Prompt]中的要求一致？（即没有明显的偏移或错位）
    - '文字排布一致性': [生成的图像]中包含的文字的排布、间距是否合理？（即没有明显的挤压、过大间距、过小间距，单词或者句子内的每个文字间距基本一致，符合常规阅读习惯）
    
    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
        "score": {
            "专项维度评估": { "语义一致性": <SCORE>, "文字数量正确性": <SCORE>, "文字内容正确性": <SCORE>, "文字可读性": <SCORE>, "文字位置正确性": <SCORE>, "文字排布一致性": <SCORE> }
        },
        "reasoning": ...
    }
    """,

    # ### 20. prompt改写任务
    "Reproduce_4oUI": """
    现在你是一个Prompt改写专家，您将收到[文本Prompt]。[文本Prompt]是关于“图像生成”的，[文本Prompt]会要求生成的图像中包含某些特定的文字内容，现在我需要你改写这些文字内容，与原来的文字内容完全不同。
    请根据以下要求进行改写：
    1. 改写后的Prompt要求生成的文字内容与原本的文字内容不同，但是数量基本一致。
    2. 必要时可以改写一些非文字内容，以保证语义的连贯。
    
    请您直接输出改写后的Prompt，**不要**添加任何额外的解释。

    给定的[文本Prompt]为：<INSTRUCTION_PLACEHOLDER>
    """,

    ### 21. Produce Long Text Bench
    "Produce_LongTextBench": """
    现在你是一个世界顶级的Prompt生成专家，您将收到一个[Prompt]以及该[Prompt]所属的[Category]作为参考。[Prompt]是关于“图像生成“的，[Prompt]会要求生成的图像中包含某些特定的文字内容，现在我需要你仿照这个[Prompt]的形式，生成一个新的Prompt。
    请根据以下要求进行生成：
    1. 生成的Prompt也属于给定的[Category]。
    2. 生成的Prompt包含的文字内容与原本的[Prompt]中的文字内容不同，且数量在40个汉字或字母以上。
    3. 生成的Prompt表述方式、语义、场景与原本的[Prompt]不同。
    4. 生成的Prompt的长度与原本的[Prompt]的长度基本相同。
    5. 生成的Prompt的语言（中文或者英文）与原本的[Prompt]的语言一致。
    6. 生成的Prompt包含的文字内容需要使用引号(中文使用“”,英文使用""）进行标注，其他的内容不要使用引号(“”和"")标注。
    7. 生成的Prompt包含的文字内容包含5-10个句子或sentences，每个句子或者sentences分开使用引号标注，这些句子或者sentences不要在生成的Prompt中连续出现，要分散在图片的不同位置。
    8. 中文引号“”包裹的内容中不要再含有中文引号“”或英文引号""，英文引号""包裹的内容中不要再含有英文引号""或中文引号“”。

    请您直接输出生成后的Prompt，**不要**添加任何额外的解释。

    给定的[Category]和[Prompt]为：<INSTRUCTION_PLACEHOLDER>
    """,

    ### 22. 畸形检测任务
    "Deformation_Detection_Original": """
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“图像生成”任务生成的图像是否有人物或其他物体畸形。您将收到[生成的图像]和[文本Prompt]。[文本Prompt]会要求[生成的图像]中包含某些特定的文字内容，可能是中文也可能是英文，您需要特别注意。
    请根据以下问题进行评估，1代表“不存在畸形”，0代表“存在畸形”。
    
    **专项维度评估**
    - '变形检测': 生成的图像中是否有人物或其他主要物体畸形？（即检测是否有人物或其他物体变形、扭曲、拉伸等，比如人脸、人的肢体存在畸形或者脸部是糊的等情况，尤其需要注意画面中的小人脸或者人手、脚等是否有畸形存在）
    
    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
        "score": <SCORE>,
        "reasoning": ...
    }
    """,

    ### 23. 畸形检测任务-优化版
    "Deformation_Detection": """
    您是一个AI图像风格质量评估系统。您的任务是根据以下标准，评估一次“图像生成”任务生成的图像的风格质量。您将收到[生成的图像]和[文本Prompt]。
    请根据以下问题进行评估，1代表**生成图像的风格与prompt要求的一致**，0代表**生成图像的风格与prompt要求的不一致**，风格比如写实、摄影、吉卜力、赛博朋克、水彩风格、油画、日本动漫、国风等等。
    
    **专项维度评估**
    - '风格一致性': 生成的图像与[文本Prompt]中要求的艺术风格是否一致？比如写实、摄影、吉卜力、赛博朋克、水彩风格、油画、日本动漫、国风等等，得分划分为[0,1,2]分共三档，0分代表完全不一致，2分代表完全一致
    
    您的最终输出必须是一个单独的、严格遵循以下格式的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
        "score": <SCORE>,
        "reasoning": ...
    }
    """,

    ### 24. 综合评估任务
    "Zonghe_Evaluation": """
    您是一个AI图像质量评估系统。您的任务是根据以下标准，评估一次“图像生成”任务生成的图像的质量。您将收到[生成的图像]和[文本Prompt]。
    请根据以下问题进行评估，除特定维度要求的评分等级外（如图像aigc感、风格一致性），1代表“质量好/成功/好/不存在畸形/不存在非prompt要求的额外文字”等正面评价，0代表“质量差/失败/差/存在畸形/存在非prompt要求的额外文字”等负面评价。
    主要从以下维度进行评价：

    - '文本与图像主体语义一致性': 生成的图像与[文本Prompt]中要求的主体是否一致，不能有缺失或者多余的主体
    - '文本与图像属性语义一致性': 生成的图像与[文本Prompt]中要求的主体属性是否一致，例如颜色、形状、位置等、以及属性是否能对应正确的主体
    - '文本与图像动作语义一致性': 生成的图像与[文本Prompt]中要求的主体动作是否一致。如果没有动作要求，则认为一致
    - '图像aigc感': 评估生成图像的油腻感（除非prompt要求的油腻感）、aigc感，得分划分为[0,1,2,3]分共四档，0分代表非常油腻、aigc感很重，3分代表完全不油腻、无aigc感
    - '风格一致性': 生成的图像与[文本Prompt]中要求的艺术风格是否一致？比如写实、摄影、吉卜力、赛博朋克、水彩风格等等，得分划分为[0,1,2,3]分共四档，0分代表完全不一致，3分代表完全一致
    - '畸形检测': 生成的图像中是否存在畸形，比如人脸、肢体、手脚、动物等是否存在畸形或着非prompt要求的模糊，尤其注意画面中的小人脸或者人手、脚等是否有畸形存在
    - '文字生成质量': 如果要求生成文字，则需要评估文字的生成质量，包括文字的准确度、排布合理性等
    - '是否存在非prompt要求的额外文字': 生成的图像中是否存在非prompt要求的额外文字，尤其注意是否有多余的文字、或难以分辨的类文字内容

    您的最终输出必须是一个单独的、严格遵循以下格式（顺序不可改变）的JSON对象，包括得分score和简洁的推理过程reasoning：
    {
        "score": {
            "文本与图像主体语义一致性": <SCORE>,
            "文本与图像属性语义一致性": <SCORE>,
            "文本与图像动作语义一致性": <SCORE>,
            "图像质量": <SCORE>,
            "风格一致性": <SCORE>,
            "畸形检测": <SCORE>,
            "文字生成质量": <SCORE>,
            "是否存在非prompt要求的额外文字": <SCORE>,
        },
        "reasoning": ...
    }
    """,
    
    ### 25. Produce CoT to train Reward Models v0
    "Produce_CoTForRM_v0": """
    [Prompt]: <PROMPT_PLACEHOLDER>
    <TASK_INSTRUCTION_PLACEHOLDER>
    <SYSTEM_PROMPT_PLACEHOLDER>
    Now I'll tell you the answer is: **<RESPONSE_PLACEHOLDER>**, please generate a concise reasoning process to support this answer.
    Please provide the concise reasoning process. **Do not** include the answer.
    """,

    ### 26. Translate Chinese to English
    "Translate_Ch_to_En": """
    You are a professional translator. Your task is to translate the following Chinese text into English while preserving its original meaning and context.
    If the text is in English, you should return it unchanged.
    If the text is in Chinese, please provide the English translation without any additional explanations.

    [Text]: <INSTRUCTION_PLACEHOLDER>
    """

}

ALL_TASK_NAMES = list(SYSTEM_PROMPTS.keys())
SCORE_WEIGHTS = {
    "Editing": [0.1, 0.02],  # 编辑准确度/过度编辑的权重
    "Editing_Subtask": 0.2,  # 总分数 0-5 -> 0-1, 如果指令跟随没得分，那总分直接为0
    "Subject_Driven_Consistency": [0.25, 0.25, 0.25, 0.15, 0.1],
    "Semantic_Alignment": [0.25, 0.15, 0.2, 0.2, 0.2],
    "OCR_3": [0.247, 0.167, 0.167, 0.127, 0.167, 0.127],
    "Zonghe_Evaluation": [0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1],  # 总分 1.4, 风格和畸形占比最高
}

EDITING_SUBTASK_NAMES_MAP = {
    "主体替换": "Editing_Subject_Replacement",
    "属性修改": "Editing_Subject_Attribute_Modification",
    "主体删除": "Editing_Subject_Deletion",
    "主体添加": "Editing_Subject_Addition",
    "动作修改": "Editing_Action_Modification",
    "场景修改": "Editing_Scene_Modification",
    "风格修改": "Editing_Style_Modification",
    "文本编辑": "Editing_Text_Modification",
    "图像修复": "Editing_Image_Restoration",
    "光照修改": "Editing_Lighting_Modification",
}

class GoogleGeminiRewardModel(object):
    def __init__(
            self, 
            app_id, 
            app_key,
            logger, 
            COS_BASE_URL: str = "",
            COS_SECRET_ID: str = "",
            COS_SECRET_KEY: str = "",
            api_version="v2.5", 
            task_name="COUNT_CN_NO_REASON", 
            score_weights=None, 
            model_marker: Optional[str] = None,
    ):
        super(GoogleGeminiRewardModel, self).__init__()
        self.app_id = app_id
        self.app_key = app_key
        self.api_version = api_version
        self.logger = logger
        if "Editing_Subtask" not in task_name:
            self.system_prompt = SYSTEM_PROMPTS[task_name]
        self.task_name = task_name
        self.model_marker = model_marker or "api_google_gemini-2.5-pro"
        if ("Editing" in self.task_name and "Editing_Instruction_Classification" not in self.task_name) or \
            ("Subject_Driven_Consistency" in self.task_name) or \
            ("Semantic_Alignment" in self.task_name) or \
            ("OCR_3" in self.task_name) or \
            ("Zonghe_Evaluation" in self.task_name):
            if score_weights is not None:
                self.score_weights = score_weights
            else:
                self.score_weights = SCORE_WEIGHTS[task_name]

        self.COS_BASE_URL = COS_BASE_URL
        self.COS_SECRET_ID = COS_SECRET_ID
        self.COS_SECRET_KEY = COS_SECRET_KEY
    
    def _reset_proxy(self):
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
    
    def parse_json_response(self, response_text):
        if response_text.startswith('```json'):
            # Extract JSON from markdown code block
            json_start = response_text.find('```json') + 7
            json_end = response_text.rfind('```')
            if json_end > json_start:
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text[json_start:].strip()
        elif response_text.startswith('```'):
            # Extract JSON from generic code block
            json_start = response_text.find('```') + 3
            json_end = response_text.rfind('```')
            if json_end > json_start:
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text[json_start:].strip()
        else:
            # Direct JSON response
            json_text = response_text
        
        return json.loads(json_text)


    def parse_response(self, response):
        if self.task_name == "Editing":
            # {
            #   "score" : [...],
            #   "reasoning" : "..."
            # }
            try:
                success = 1
                parsed_response = self.parse_json_response(response[0])
                score = parsed_response["score"]
                assert len(score) == 2, f"score must be a list of two elements, but got {score}"
                score = score[0] * self.score_weights[0] + score[1] * self.score_weights[1]
                reasoning = parsed_response["reasoning"]
            except Exception as e:
                self.logger.error(f"parse response error: {e}, response text: {response[0]}")
                return 0, 0, ""
            return success, score, reasoning
        elif "Editing_Subtask" in self.task_name:
            success = 0
            # score: {
            # "通用维度评估": { "指令遵循度": <SCORE>, "技术质量": <SCORE>, "非编辑区稳定性": <SCORE>},
            # "专项维度评估": { "光照效果真实性": <SCORE>, "光影逻辑一致性": <SCORE>, "细节保持度": <SCORE> }
            # }
            try:
                parsed_response = self.parse_json_response(response[0])
                score_json = parsed_response["score"]
                total_score = 1.0
                scores_general = 0
                scores_special = 0
                # 总分为5，专项得分总分2，通用 3, 如果指令遵循度没得分，那直接总得分为0
                weight_special = 2.0 / len(score_json["专项维度评估"])
                for key, value in score_json.items():
                    if key == "通用维度评估":
                        for sub_key, sub_value in value.items():
                            if sub_key == "指令遵循度" and sub_value < 0.1:
                                total_score = 0
                            scores_general += sub_value
                    elif key == "专项维度评估":
                        for sub_key, sub_value in value.items():
                            scores_special += sub_value * weight_special
                if total_score > 0:
                    total_score = (scores_general + scores_special) * self.score_weights
                reasoning = parsed_response["reasoning"]
                return 1, total_score, reasoning
            except Exception as e:
                self.logger.error(f"parse response error: {e}, response text: {response[0]}")
                return 0, 0, ""
        elif "Semantic_Alignment" in self.task_name:
            success = 0
            try:
                parsed_response = self.parse_json_response(response[0])
                score_json = parsed_response["score"]
                total_score = 0.0
                scores_special = []
                for key, value in score_json.items():
                    if key == "专项维度评估":
                        for sub_key, sub_value in value.items():
                            scores_special.append(sub_value)
                for i, score_special in enumerate(scores_special):
                    total_score += score_special * self.score_weights[i]
                reasoning = parsed_response["reasoning"]
                return 1, total_score, reasoning
            except Exception as e:
                self.logger.error(f"parse response error: {e}, response text: {response[0]}")
                return 0, 0, ""
        elif "Subject_Driven_Consistency" in self.task_name:
            success = 0
            try:
                parsed_response = self.parse_json_response(response[0])
                score_json = parsed_response["score"]
                total_score = 0.0
                scores_general = []
                for key, value in score_json.items():
                    if key == "通用维度评估":
                        for sub_key, sub_value in value.items():
                            scores_general.append(sub_value)
                for i, score_general in enumerate(scores_general):
                    total_score += score_general * self.score_weights[i]
                reasoning = parsed_response["reasoning"]
                return 1, total_score, reasoning
            except Exception as e:
                self.logger.error(f"parse response error: {e}, response text: {response[0]}")
                return 0, 0, ""
        elif "OCR_3" in self.task_name:
            success = 0
            try:
                parsed_response = self.parse_json_response(response[0])
                score_json = parsed_response["score"]
                total_score = 0.0
                scores_special = []
                for key, value in score_json.items():
                    if key == "专项维度评估":
                        for sub_key, sub_value in value.items():
                            scores_special.append(sub_value)
                for i, score_special in enumerate(scores_special):
                    total_score += score_special * self.score_weights[i]
                reasoning = parsed_response["reasoning"]
                return 1, total_score, reasoning
            except Exception as e:
                self.logger.error(f"parse response error: {e}, response text: {response[0]}")
                return 0, 0, ""
        elif "Deformation_Detection" in self.task_name:
            success = 0
            try:
                parsed_response = self.parse_json_response(response[0])
                score = parsed_response["score"]
                reasoning = parsed_response["reasoning"]
                return 1, score, reasoning
            except Exception as e:
                self.logger.error(f"parse response error: {e}, response text: {response[0]}")
                return 0, 0, ""
        elif "Zonghe_Evaluation" in self.task_name:
            success = 0
            try:
                parsed_response = self.parse_json_response(response[0])
                score_json = parsed_response["score"]
                total_score = 0.0
                for i, value in enumerate(score_json.values()):
                    total_score += value * self.score_weights[i]
                reasoning = parsed_response["reasoning"]
                return 1, total_score, reasoning
            except Exception as e:
                self.logger.error(f"parse response error: {e}, response text: {response[0]}")
                return 0, 0, ""
        else:
            success = 0
            score = 0.0
            if "正确" in response[0]:
                success = 1
                score = 1.0
            elif "错误" in response[0]:
                success = 1
                score = 0.0
            return success, score
    
    def editing_instruction_classification(self, prompt, gemini_try_times: int = 2, timeout: int = 30):
        if prompt is None:
            return None
        self._reset_proxy()
        system_prompt = SYSTEM_PROMPTS["Editing_Instruction_Classification"].replace("<INSTRUCTION_PLACEHOLDER>", prompt)
        response = request_batch(
            images=None,
            system_prompt=system_prompt,
            user_prompt=prompt,
            app_id=self.app_id,
            app_key=self.app_key,
            api_version=self.api_version,
            model_marker=self.model_marker,
            COS_BASE_URL=self.COS_BASE_URL,
            COS_SECRET_ID=self.COS_SECRET_ID,
            COS_SECRET_KEY=self.COS_SECRET_KEY,
        )
        category = response[0]

        for key, value in EDITING_SUBTASK_NAMES_MAP.items():
            if key in category:
                return value
        
        raise ValueError(f"Unknown category: {category}")
    
    def t2i_prompt_semantic_decomposition(self, prompt):
        system_prompt = SYSTEM_PROMPTS["T2I_Prompt_Semantic_Decomposition"].replace("<INSTRUCTION_PLACEHOLDER>", prompt)
        response = request_batch(
            images=None,
            system_prompt=system_prompt,
            user_prompt=prompt,
            app_id=self.app_id,
            app_key=self.app_key,
            api_version=self.api_version,
            model_marker=self.model_marker,
        )
        return response[0]

    def reproduce_4oUI_prompt_rewrite(self, prompt, try_times: int = 3, timeout: int = 30):
        system_prompt = SYSTEM_PROMPTS["Reproduce_4oUI"].replace("<INSTRUCTION_PLACEHOLDER>", prompt)
        response = request_batch(
            images=None,
            system_prompt=system_prompt,
            user_prompt=prompt,
            app_id=self.app_id,
            app_key=self.app_key,
            api_version=self.api_version,
            model_marker=self.model_marker,
            gemini_try_times=try_times,
            timeout=timeout,
        )
        return response[0]
    
    def produce_LongTextBench_prompt(self, prompt, try_times: int = 3, timeout: int = 30):
        system_prompt = SYSTEM_PROMPTS["Produce_LongTextBench"].replace("<INSTRUCTION_PLACEHOLDER>", prompt)
        response = request_batch(
            images=None,
            system_prompt=system_prompt,
            user_prompt=prompt,
            app_id=self.app_id,
            app_key=self.app_key,
            api_version=self.api_version,
            model_marker=self.model_marker,
            gemini_try_times=try_times,
            timeout=timeout,
        )
        return response[0]

    def translate_ch_en(self, prompt, try_times: int = 3, timeout: int = 30):
        system_prompt = SYSTEM_PROMPTS["Translate_Ch_to_En"].replace("<INSTRUCTION_PLACEHOLDER>", prompt)
        response = request_batch(
            images=None,
            system_prompt=system_prompt,
            user_prompt=None,
            app_id=self.app_id,
            app_key=self.app_key,
            api_version=self.api_version,
            model_marker=self.model_marker,
            gemini_try_times=try_times,
            timeout=timeout,
        )
        return response[0]

    def eval(
        self, 
        images: List[Image.Image],
        prompts: Union[str, List[str]],
        src_images: Optional[List[Image.Image]] = None,
        max_workers: int = 1,
        show_progress: bool = False,
        app_id: str = None,
        app_key: str = None,
        put_img_to_cos: bool = False,
        resize_images: bool = False,
        gemini_try_times: int = 30,
        timeout: int = 3600,
    ) -> Union[tuple[list, list], tuple[list, list, list]]:
        self._reset_proxy()
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError("prompts must have the same length as images")
        
        scores = []
        successes = []
        reasonings = []
        app_id = app_id if app_id is not None else self.app_id
        app_key = app_key if app_key is not None else self.app_key
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            if resize_images:
                if max(image.size) > 512:
                    ratio = 512 / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            if src_images is not None:
                src_image = src_images[i]
                if resize_images:
                    if max(src_image.size) > 512:
                        ratio = 512 / max(src_image.size)
                        new_size = tuple(int(dim * ratio) for dim in src_image.size)
                        src_image = src_image.resize(new_size, Image.Resampling.LANCZOS)
            
            if "Editing_Subtask" in self.task_name:
                subtask_name = self.editing_instruction_classification(prompt)
                self.logger.info(f"Current editing subtask: {subtask_name}")
                system_prompt = SYSTEM_PROMPTS[subtask_name].replace("<INSTRUCTION_PLACEHOLDER>", prompt)
                prompt = ""
            elif "Semantic_Alignment" in self.task_name:
                system_prompt = self.system_prompt
                prompt = "[Prompt语义拆分内容]: " + prompt
                
            elif "Editing" in self.task_name or \
                 "Subject_Driven_Consistency" in self.task_name:
                system_prompt = self.system_prompt.replace("<INSTRUCTION_PLACEHOLDER>", prompt)
                prompt = ""
            
            elif "OCR_3" in self.task_name:
                system_prompt = self.system_prompt
                prompt = "[文本Prompt]: " + prompt

            elif "Zonghe_Evaluation" in self.task_name:
                system_prompt = SYSTEM_PROMPTS["Zonghe_Evaluation"]  #.replace("<INSTRUCTION_PLACEHOLDER>", prompt)
            
            elif "Deformation_Detection" in self.task_name:
                system_prompt = SYSTEM_PROMPTS["Deformation_Detection"]  #.replace("<INSTRUCTION_PLACEHOLDER>", prompt)

            else:
                system_prompt = self.system_prompt

            response = request_batch(
                images=[image] if src_images is None else [[src_image, image]],
                system_prompt=system_prompt,
                user_prompt=prompt,
                app_id=app_id,
                app_key=app_key,
                api_version=self.api_version,
                model_marker=self.model_marker,
                max_workers=max_workers,
                show_progress=show_progress,
                put_img_to_cos=put_img_to_cos,
                COS_BASE_URL=self.COS_BASE_URL,
                COS_SECRET_ID=self.COS_SECRET_ID,
                COS_SECRET_KEY=self.COS_SECRET_KEY,
                gemini_try_times=gemini_try_times,
                timeout=timeout,
            )
            if self.logger is not None:
                self.logger.info(f"response: {response}")
            
            if "Editing" in self.task_name or \
                "Subject_Driven_Consistency" in self.task_name or \
                "Semantic_Alignment" in self.task_name or \
                "OCR_3" in self.task_name or \
                "Deformation_Detection" in self.task_name or \
                "Zonghe_Evaluation" in self.task_name:
                success, score, reasoning = self.parse_response(response)
                # logger.info(f"score: {score}, success: {success}, reasoning: {reasoning}")
                reasonings.append(reasoning)
            else:
                success, score = self.parse_response(response)
            
            scores.append(score)
            successes.append(success)

        return scores, successes

class SubDriCons_RM_Face(FaceSimilarityRewardModel):
    def __init__(self, http_proxy=None, https_proxy=None):
        super().__init__(
            http_proxy=http_proxy, 
            https_proxy=https_proxy
        )

    def eval(
        self, 
        images: List[Image.Image], 
        reference_images: List[Image.Image], 
        use_face_rewards: List[bool]
    ):
        assert len(images) == len(reference_images) == len(use_face_rewards), \
            "images, reference_images, and use_face_rewards must have the same length"

        # 初始化 rewards 和 successes
        rewards = [0.0] * len(images)  # 默认 reward 为 0.0
        successes = [1] * len(images)  # 默认 success 为 True

        # 分组：提取需要计算的样本
        indices_to_compute = [i for i, use_face in enumerate(use_face_rewards) if use_face]
        images_to_compute = [images[i] for i in indices_to_compute]
        reference_images_to_compute = [reference_images[i] for i in indices_to_compute]

        # 批量调用父类 __call__ 方法
        if indices_to_compute:
            batch_rewards, batch_successes = super().__call__(
                images=images_to_compute, reference_images=reference_images_to_compute
            )

            # 将批量计算结果填回原始位置
            for idx, reward, success in zip(indices_to_compute, batch_rewards, batch_successes):
                rewards[idx] = reward
                successes[idx] = success

        return rewards, successes


class SubDriCons_RM_Gemini(GoogleGeminiRewardModel):
    def __init__(
        self, 
        app_id, 
        app_key, 
        logger, 
        api_version="v2.5", 
        score_weights=None, 
        model_marker=None
    ):
        super().__init__(
            app_id=app_id, 
            app_key=app_key, 
            logger=logger, 
            api_version=api_version, 
            task_name="Subject_Driven_Consistency", 
            score_weights=score_weights, 
            model_marker=model_marker
        )

    def eval(
        self, 
        images: List[Image.Image], 
        prompts: List[str], 
        src_images: List[Image.Image], 
        use_face_rewards: List[bool], 
        max_workers: int = 1, 
        show_progress: bool = False, 
        app_id: str = None, 
        app_key: str = None, 
        put_img_to_cos: bool = False, 
        resize_images: bool = True
    ):
        # 校验长度一致
        assert len(images) == len(src_images) == len(use_face_rewards), "images, src_images and use_face_rewards must have the same length"

        # 如果 prompts 是字符串，转换成列表
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        else:
            assert len(prompts) == len(images), "prompts length must match images length"

        # 初始化 rewards 和 successes
        rewards = [0.0] * len(images)  # 默认 reward 为 0
        successes = [1] * len(images)  # 默认 success 为 True

        # 根据 use_face_rewards 分组
        indices_to_compute = [i for i, use_face in enumerate(use_face_rewards) if not use_face]

        # 如果有需要计算的样本
        if indices_to_compute:
            # 批量提取需要计算的样本
            images_to_compute = [images[i] for i in indices_to_compute]
            prompts_to_compute = [prompts[i] for i in indices_to_compute]
            src_images_to_compute = [src_images[i] for i in indices_to_compute]

            # 批量调用父类 eval
            batch_rewards, batch_successes = super().eval(
                images=images_to_compute, 
                prompts=prompts_to_compute, 
                src_images=src_images_to_compute, 
                max_workers=max_workers, 
                show_progress=show_progress, 
                app_id=app_id, 
                app_key=app_key, 
                put_img_to_cos=put_img_to_cos, 
                resize_images=resize_images
            )

            # 将计算结果填入 rewards 和 successes
            for idx, reward, success in zip(indices_to_compute, batch_rewards, batch_successes):
                rewards[idx] = reward
                successes[idx] = success

        # 返回批量处理后的结果
        return rewards, successes

class SemAlign_RM(GoogleGeminiRewardModel):
    def __init__(
            self, 
            app_id, 
            app_key, 
            logger, 
            api_version="v2.5",
            score_weights=None, 
            model_marker: Optional[str] = None
    ):
        super(SemAlign_RM, self).__init__(
            app_id=app_id, 
            app_key=app_key, 
            logger=logger, 
            api_version=api_version, 
            task_name="Semantic_Alignment", 
            score_weights=score_weights, 
            model_marker=model_marker
        )

class GG_OCR3(GoogleGeminiRewardModel):
    def __init__(
            self, 
            app_id, 
            app_key, 
            logger, 
            api_version="v2.5", 
            score_weights=None, 
            model_marker: Optional[str] = None
    ):
        super(GG_OCR3, self).__init__(
            app_id=app_id, 
            app_key=app_key, 
            logger=logger, 
            api_version=api_version, 
            task_name="OCR_3", 
            score_weights=score_weights, 
            model_marker=model_marker,
        )

class EditingCons_RM_Gemini(GoogleGeminiRewardModel):
    def __init__(
        self,
        app_id,
        app_key,
        logger,
        api_version="v2.5",
        score_weights=None, 
        model_marker: Optional[str] = None
    ):
        super(EditingCons_RM_Gemini, self).__init__(
            app_id=app_id, 
            app_key=app_key, 
            logger=logger, 
            api_version=api_version, 
            task_name="Editing_Subtask", 
            score_weights=score_weights, 
            model_marker=model_marker,
            max_workers=4,
        )
