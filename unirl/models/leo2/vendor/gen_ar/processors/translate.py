import argparse
import json
import os
from functools import partial
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

GPT4_ENGINE = os.getenv("GPT4_ENGINE", None)
GPT4_TOKEN = os.getenv("GPT4_TOKEN", None)
GPT4_URL = os.getenv("GPT4_URL", None)

QWEN_URL = os.getenv("QWEN_URL", None)

zh2en_template = """
你是一个将中文翻译成英文的翻译专家,现在给定下面中文文本，需要你将其翻译成地道的英文.
1、注意不要死板按词翻译, 翻译过程要注意中英文之间专业术语对应关系, 中文人名的翻译要合理。
2、注意中文表达顺序和英语表达顺序的习惯和区别, 要翻译成符合英文社区常见的地道表达。

#待翻译中文文本:

{}

#地道英文翻译结果（只输出英文翻译结果，回答不要包含其他语言）:
"""

en2zh_template = """
你是一个将英文翻译成中文的翻译专家,现在给定下面英文文本，需要你将其翻译成中文.
1、注意不要死板按词翻译, 翻译过程要注意中英文之间专业术语对应关系, 中文人名的翻译要合理。
2、注意中文表达顺序和英语表达顺序的习惯和区别, 要翻译成符合中文社区常见的地道表达。

#待翻译英文文本:

{}

#地道中文翻译结果（只输出中文翻译结果，回答不要包含其他语言）:
"""


def parse_results(resp):
    data = json.loads(resp.text)
    try:
        content = data["choices"][0]["message"]["content"]
    except KeyError as e:
        raise KeyError(f"Failed to parse response with '{e}', response: {data}")
    return content


# 翻译prompt成英文
def gpt_translate(sentence, template, try_times=5):
    if not GPT4_ENGINE or not GPT4_TOKEN or not GPT4_URL:
        raise ValueError("Please set GPT4_ENGINE, GPT4_TOKEN and GPT4_URL in environment variables.")

    prompt = template.format(sentence)
    data = {
        "engine": GPT4_ENGINE,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    headers = {"Authorization": f'Bearer {GPT4_TOKEN}', "Content-Type": "json"}

    content = ""
    for _ in range(try_times):
        try:
            resp = requests.post(
                GPT4_URL,
                data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
                headers=headers,
            )
            content = parse_results(resp)
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to translate with {type(e)}: '{e}', retrying...")

    return content


def tencent_translate(text, source, target, try_times=1):
    TENCENT_TRANSLATE_URL = os.getenv("TENCENT_TRANSLATE_URL", None)
    TENCENT_TRANSLATE_SERVICE = os.getenv("TENCENT_TRANSLATE_SERVICE", None)
    if TENCENT_TRANSLATE_URL is None or TENCENT_TRANSLATE_SERVICE is None:
        raise ValueError("Please set TENCENT_TRANSLATE_URL and TENCENT_TRANSLATE_SERVICE in environment variable")
    trans_text = ""
    for _ in range(try_times):
        try:
            req_temp = {
                "business": TENCENT_TRANSLATE_SERVICE,
                "interface": 1998,
                "text": text,
                "source_lang": source,
                'target_lang': target,
            }
            results = requests.post(url=TENCENT_TRANSLATE_URL, json=req_temp).json()
            print(results)
            trans_text = results['translatedText']
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to translate with error '{e}', retrying...")
    return trans_text


def qwen_translate(sentence, template, try_times=5):
    if QWEN_URL is None:
        raise ValueError("Please set QWEN_URL in environment variables.")

    prompt = template.format(sentence)
    headers = {"Content-Type": "application/json"}

    data = {
        "model": "/mnt/model/vllm-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "top_p": 0.3,
        "top_k": 50
    }

    content = ""
    for _ in range(try_times):
        try:
            response = requests.post(QWEN_URL, headers=headers, data=json.dumps(data))
            content = response.json()['choices'][0]['message']['content']
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to translate with {type(e)}: '{e}', retrying...")

    return content


def read_file(src, text_col, trans_col):
    src_file = Path(src)
    df = pd.read_csv(src_file, header=0)
    logger.info(f"Read {len(df)} records from {src_file}")

    # Check text column
    if text_col not in df.columns:
        raise ValueError(f"Column {text_col} not found in the file")
    if trans_col in df.columns:
        logger.warning(f"Column {trans_col} already exists in the file, will be overwritten")

    return df


def get_translate_model(model, lang):
    if model == "gpt4":
        if lang == 'zh-en':
            translate_func = partial(gpt_translate, template=zh2en_template)
        elif lang == 'en-zh':
            translate_func = partial(gpt_translate, template=en2zh_template)
        else:
            raise ValueError(f"Invalid language: {lang}")
    elif model == "tencent":
        if lang == 'zh-en':
            translate_func = partial(tencent_translate, source='zh', target='en')
        elif lang == 'en-zh':
            translate_func = partial(tencent_translate, source='en', target='zh')
        else:
            raise ValueError(f"Invalid language: {lang}")
    elif model == "qwen":
        if lang == 'zh-en':
            translate_func = partial(qwen_translate, template=zh2en_template)
        elif lang == 'en-zh':
            translate_func = partial(qwen_translate, template=en2zh_template)
        else:
            raise ValueError(f"Invalid language: {lang}")
    else:
        raise ValueError(f"Invalid model: {model}")
    return translate_func


def process_file(src, text_col, trans_col, model, dst, lang, cut=False):
    df = read_file(src, text_col, trans_col)
    # Add new column
    df[trans_col] = ""

    translate_func = get_translate_model(model, lang)

    # Save file
    save_file = Path(dst)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = save_file.with_suffix('.tmp.csv')

    # Skip finished rows
    if tmp_file.exists():
        tmp_df = pd.read_csv(tmp_file, header=0, keep_default_na=False)
        # Get indices that trans_col is empty string
        tmp_df = tmp_df[tmp_df[trans_col] == ""]
        indices = tmp_df.index
        logger.info(f"Skip {len(df) - len(indices)} finished records. Continue to translate {len(indices)} records.")

        df = df.loc[indices]

    # Start translation
    pbar = tqdm(enumerate(df.iterrows()), total=len(df))
    for i, (index, row) in pbar:
        prompt = row[text_col]
        # If the prompt is NaN or empty string, skip the current iteration
        if pd.isna(prompt) or prompt == "":
            continue
        translated_prompt = translate_func(prompt, try_times=3)
        if translated_prompt.strip():
            if cut:
                df.loc[index, trans_col] = str(translated_prompt).split('\n')[0].strip()
            else:
                df.loc[index, trans_col] = str(translated_prompt).strip()

        pbar.set_description(f"{prompt[:40]:<40s} -> {translated_prompt[:40]:<40s}")

        if i % 10 == 0:
            # Save to a temporary file
            df.to_csv(tmp_file, index=False)

    # Save to the final file
    df.to_csv(save_file, index=False)
    if save_file.exists():
        logger.info(f"Save to {save_file} with {len(df)} records")
        # Remove the temporary file
        if tmp_file.exists():
            tmp_file.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description='Translate text from Chinese to English or English to Chinese')
    parser.add_argument('model', type=str, choices=['gpt4', 'test_gpt4', 'tencent', 'test_tencent', 'qwen', 'test_qwen'],
                        help='The translation model')
    parser.add_argument('-s', '--src', type=str, help='The text file to be translated')
    parser.add_argument('-t', '--dst', type=str, help='The target file')
    parser.add_argument('-c', '--col', type=str, help='The column name of the text')
    parser.add_argument('-l', '--lang', type=str, default=None, choices=['zh-en', 'en-zh'],
                        help='The language of the text file')
    parser.add_argument('--cut', action='store_true', help='Cut the translation result by the first line')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model == 'test_gpt4':
        print(gpt_translate('你好啊', zh2en_template))
    elif args.model == 'test_tencent':
        print(tencent_translate('你好啊', 'zh', 'en'))
    elif args.model == 'test_qwen':
        print(qwen_translate('你好啊', zh2en_template))
    else:
        text_col, trans_col = args.col.split('->')
        process_file(args.src,
                     text_col,
                     trans_col,
                     args.model,
                     args.dst,
                     args.lang,
                     args.cut,
                     )


if __name__ == '__main__':
    main()


# source /apdcephfs_nj8/share_301739632/jarvizhang/workspace/hunyuan_multimodal/api_key.sh
# python3 processors/translate.py qwen -s data/test/editing_HIVE.csv -t data/test/editing_HIVE_ml.csv -c 'prompt->prompt_cn'  -l en-zh --cut
