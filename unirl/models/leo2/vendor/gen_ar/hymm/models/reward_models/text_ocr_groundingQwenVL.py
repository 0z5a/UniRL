import pandas as pd
import json
import re
import base64
import os
from hymm.models.reward_models.utils.request_service import request_batch, request_one
from hymm.models.reward_models.utils.ocr_rw_rule import calculate_metrics, get_matches_info


class TextOCRGroundingQwenVL(object):
    def __init__(self):
        super(TextOCRGroundingQwenVL, self).__init__()
    
    def _reset_proxy(self):
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)

    def eval(self, imgs_pil, gts, url, max_workers=8, show_progress=False, if_split_by_character=False, **kwargs):
        # Reset proxy, otherwise cannot access the server url
        self._reset_proxy()
        assert isinstance(imgs_pil, list)
        samples = [
            {"image": img_pil}
            for img_pil in imgs_pil
        ]
        results = request_batch(url, samples, max_workers=max_workers, show_progress=show_progress)
 
        METRICS_KEYS = ['precision', 'recall', 'f1_score', 'matched_count', 'gt_count', 'pred_count']
        eval_results = []
        for result, gt in zip(results, gts):
            eval_result = {}
            eval_result['pred_ocr'] = result['pred_ocr']
            eval_result['pred_ocr_json'] = result['pred_ocr_json']
            if 'error' in result:
                for key in METRICS_KEYS:
                    eval_result[key] = -100
                continue

            pred_ocr = json.loads(result['pred_ocr_json'])
            gt_ocr = [self.filter_str(ocr) for ocr in gt]
            pred_ocr = [self.filter_str(str(ocr)) for ocr in pred_ocr]

            if if_split_by_character:
                gt_ocr_split = []
                for ocr in gt_ocr:
                    parts = self.split_by_character(ocr)
                    gt_ocr_split.extend(parts)
                pred_ocr_split = []
                for ocr in pred_ocr:
                    parts = self.split_by_character(ocr)
                    pred_ocr_split.extend(parts)
                matches_info = get_matches_info(gt_ocr_split, pred_ocr_split)
                eval_result['gt_ocr_split1'] = gt_ocr_split
                eval_result['pred_ocr_split1'] = pred_ocr_split
                metrics = calculate_metrics(gt_ocr_split, pred_ocr_split, matches_info['matches'])
            else:
                # Try two approaches: with and without splitting by punctuation
                # Approach 1: Without splitting by punctuation
                matches_info_no_split = get_matches_info(gt_ocr, pred_ocr)
                metrics_no_split = calculate_metrics(gt_ocr, pred_ocr, matches_info_no_split['matches'])
                eval_result['gt_ocr_split2'] = gt_ocr
                eval_result['pred_ocr_split2'] = pred_ocr

                # Approach 2: With splitting by punctuation
                # Split ground truth and predictions by punctuation
                gt_ocr_split = []
                for text in gt_ocr:
                    parts = self.split_by_punctuation(text)
                    gt_ocr_split.extend(parts)

                matches_info_split = get_matches_info(gt_ocr_split, pred_ocr)
                metrics_split = calculate_metrics(gt_ocr_split, pred_ocr, matches_info_split['matches'])
                eval_result['gt_ocr_split3'] = gt_ocr_split
                eval_result['pred_ocr_split3'] = pred_ocr

                # Choose the approach with higher F1 score
                metrics = metrics_split if metrics_split['f1_score'] > metrics_no_split['f1_score'] else metrics_no_split

            for key in METRICS_KEYS:
                eval_result[key] = metrics[key]

            eval_results.append(eval_result)

        return eval_results

    def parse_gt(self, prompt:str):
        assert isinstance(prompt, str)
        matches_cn = re.findall(r'“(.*?)”', prompt)
        matches_en = re.findall(r'"(.*?)"', prompt)
        matches = matches_cn + matches_en
        return matches

    def filter_str(self, line):
        line = line.strip()
        line = re.sub(r'\s+', ' ', line)
        line = line.replace('，', ',')
        line = line.replace('。', '.')
        line = line.replace('？', '?')
        line = line.replace('：', ':')
        line = line.replace('！', '!')
        Set = set(':,.? !')
        for ch in line:
            if not u'\u4e00' <= ch <= u'\u9fff' and not ch.isalnum() and ch not in Set and ch != ' ':
                line = line.replace(ch, '')
        return line    

    def split_by_character(self, text):
        """
        Split a string by characters, handling both Chinese and English text.
        For English text, split by words (spaces).
        For Chinese text, split by individual characters.

        Args:
            text (str): The input text to be split.

        Returns:
            list: A list of substrings split according to the rules.
        """
        # Check if the text contains Chinese characters
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)

        if not has_chinese:
            # For English text, split by spaces
            return self.split_by_space(text)
        else:
            # For text with Chinese characters, process differently
            result = []
            current_word = ""

            for char in text:
                # Check if the character is Chinese
                is_chinese = '\u4e00' <= char <= '\u9fff'
                # Check if the character is punctuation
                is_punctuation = re.match(r'[,.。，、?？!！;；:：]', char)
                # Check if the character is a space
                is_space = char.isspace()

                if is_chinese or is_punctuation:
                    # If we have accumulated English characters, add them as a word
                    if current_word:
                        result.append(current_word)
                        current_word = ""
                    # Add Chinese character or punctuation as a separate item
                    result.append(char)
                elif is_space:
                    # If we encounter a space, add the current English word if it exists
                    if current_word:
                        result.append(current_word)
                        current_word = ""
                else:
                    # Accumulate English characters
                    current_word += char

            # Add any remaining English word
            if current_word:
                result.append(current_word)

            # Filter out empty strings
            result = [part for part in result if part]

            return result


    def split_by_punctuation(self, text):
        """
        Split a string by punctuation marks, keeping the punctuation with the preceding text.

        Args:
            text (str): The input text to be split.

        Returns:
            list: A list of substrings split by punctuation, with punctuation included at the end of each substring.
        """
        # Define punctuation marks for splitting
        punctuation = r'([,.。，、?？!！;；:：])'

        # Split the text by punctuation but keep the punctuation
        parts = re.split(punctuation, text)

        # Combine each part with its following punctuation
        result = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(punctuation, parts[i+1]):
                combined = parts[i] + parts[i+1]
                result.append(combined.strip())
                i += 2
            else:
                if parts[i].strip():
                    result.append(parts[i].strip())
                i += 1

        # Filter out empty strings
        result = [part for part in result if part]

        return result


    def split_by_space(self, text):
        """
        Split a string by spaces.

        Args:
            text (str): The input text to be split.

        Returns:
            list: A list of substrings split by spaces.
        """
        # Split the text by spaces
        parts = text.split()

        # Filter out empty strings
        result = [part for part in parts if part]

        return result
