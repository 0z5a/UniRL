import os, re, glob, csv, sys

import paddle
import tqdm, json

paddle.utils.run_check()
paddle.device.set_device("gpu")
from paddleocr import PaddleOCR, draw_ocr
from paddleocr.paddleocr import get_model_config, parse_args
from paddleocr.tools.infer.predict_rec import TextRecognizer
from paddleocr.tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop

import torch
from torchvision import transforms as T
from PIL import Image
import os, re
import torch.nn as nn
import torch.nn.functional as F
import json
import time

import pandas as pds

import multiprocessing as mp

import numpy as np

class TextOCRGroundingPaddle(object):
    def __init__(self, device="cuda", conf_thre=0.85):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            use_gpu=True,
            device=device,
            lang="ch",
            det_model_dir="/apdcephfs_cq8/share_1367250/nanyangdu/GlyphDraw2/model_ocr/ch_PP-OCRv4_det_infer",
            rec_model_dir="/apdcephfs_cq8/share_1367250/nanyangdu/GlyphDraw2/model_ocr/ch_PP-OCRv4_rec_infer",
            cls_model_dir="/apdcephfs_cq8/share_1367250/nanyangdu/GlyphDraw2/model_ocr/ch_ppocr_mobile_v2.0_cls_infer",
            show_log=False,
        )
        self.conf_thre = 0.85

    def parse_gt(self, prompt:str):
        assert isinstance(prompt, str)
        matches = re.findall(r'“(.*?)”', prompt)
        return matches
            

    def do_ocr(self, img_array:np.ndarray):
        assert isinstance(img_array, np.ndarray)
        with torch.no_grad():
            img = Image.fromarray(img_array).convert('RGB')
            img_cv = np.array(img)
            dt_boxes, rec_res, _ = self.ocr(img_cv, cls=False) # dt_boxes:检测框，rec_res:(识别结果，置信度)

            if not rec_res:
                return None
            rec_res = [(x[0], round(x[1], 3)) for x in rec_res] # 过滤掉水印

            return dt_boxes, rec_res

    def get_ocr_acc(self, rec_res, gt, level='char'):
        acc_all = []
        for gt_i, rec_res_i in zip(gt, rec_res):
            acc = 0.0
            if level == 'image':
                acc = self.get_image_level(rec_res_i, gt_i)
            elif level == 'word':
                acc = self.get_word_level(rec_res_i, gt_i)
            elif level == 'char':
                acc = self.get_char_level(rec_res_i, gt_i)
            else:
                raise ValueError(f"level {level} not in ['image', 'word', 'char', 'all']")
            
            acc_all.append(acc)

        return acc_all
        

    def get_image_level(self, rec_res, gt):
        if len(gt) == 0:
            return None
        pred = [x[0] for x in rec_res if x[1] > self.conf_thre]
        return 1 if sorted(pred) == sorted(gt) else 0
    
    def get_word_level(self, rec_res, gt):
        pred = [x[0] for x in rec_res if x[1] > self.conf_thre]
        alls = [1 if x in pred else 0 for x in gt]
        if len(alls) == 0:
            return None

        res = round(sum(alls) / len(alls), 2)
        return res
    
    def get_char_level(self, rec_res, gt):
        pred = [x[0] for x in rec_res if x[1] > self.conf_thre] # words
        pred = [y for x in pred for y in x] # chars
        alls = [1 if x in pred else 0 for x in gt]
        if len(alls) == 0:
            return None

        res = round(sum(alls) / len(alls), 2)
        return res