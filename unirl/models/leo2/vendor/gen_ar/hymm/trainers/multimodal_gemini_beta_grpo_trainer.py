import concurrent.futures
import gc
import json
import os
import random
import time
from typing import Dict, Union, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .helpers import (
    CycleStates,
    GRPOTrainingStates,
    save_checkpoint,
)
from .multimodal_gemini_alpha_trainer import GeminiTrainerAlphaMultiModal
from ..ar.pipelines.pipeline_transfusion_text2image_with_logprob import compute_log_prob
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.rl_t2i_loader import RLTextImageArrowStream
from ..data_kits.samplers import RepeatRandomDistributedSampler
from ..models import build_model
from ..models.reward_models.altclip_rm import AltCLIPRM
from ..models.reward_models.edit_ref_rm import EditRefRM
from ..models.reward_models.editscore import EditScoreRewardModel
from ..models.reward_models.google_gemini_request import GoogleGeminiRewardModel, GG_OCR3, SubDriCons_RM_Face, SubDriCons_RM_Gemini, SemAlign_RM, EditingCons_RM_Gemini
from ..models.reward_models.hps_clip import HPSClipRewardModel
from ..models.reward_models.hpsv3_reward import HPSV3RewardModel
from ..models.reward_models.image_reward import ImageRewardModel
from ..models.reward_models.pick_score import PickScoreRewardModel
from ..models.reward_models.preference_reward_server import PreferenceRewardModel
from ..models.reward_models.text_ocr_groundingQwenVL import TextOCRGroundingQwenVL
from ..models.reward_models.unified_reward import UnifiedRewardModel
from ..samplers.gemini_beta_sampler import GeminiBetaSampler
from ..utils.deepspeed_utils import unwrap_model_for_generation_deepspeed
from ..utils.rl_exploration_metric import average_exploration_score
from ..utils.torch_utils import (
    move_model_params_and_grads_to,
    profiler_context,
    set_manual_seed,
    set_worker_seed_builder,
    PRECISION_TO_TYPE,
)
from ..utils.torch_distributions import gather_tensor

gc.set_threshold(7000, 100, 100)


class MultiModalGeminiBetaGRPOTrainer(GeminiTrainerAlphaMultiModal):
    def __init__(self, args, all_dataset_keys=None):
        super().__init__(args)
        if self.kl_weight > 0:
            self.build_reference_model()
        self.build_sampler()
    
    def set_proxy(self):
        os.environ["http_proxy"] = "http://star-proxy.oa.com:3128"
        os.environ["https_proxy"] = "http://star-proxy.oa.com:3128"
    
    def unset_proxy(self):
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""

    def build_extra_model(self):
        """ Extra frozen models. """
        self.set_proxy()
        self.reward_models = []

        ############################# Build reward models #############################
        # Qwen OCR reward model: Use the remote server to compute the reward
        if self.args.get("text_ocr", False):
            self.text_ocr_reward_model = TextOCRGroundingQwenVL()
            self.reward_models.append(self.text_ocr_reward_model)

        # AltCLIP reward model
        if self.args.get("altclip_reward", False):
            self.altclip_reward_model = AltCLIPRM(
                ft_model_path=self.args["altclip_reward"]["ft_model_path"],
                pretrained_model_name_or_path=self.args["altclip_reward"]["pretrained_model_name_or_path"],
                processor_cache_dir=self.args["altclip_reward"]["processor_cache_dir"],
                resize_res=512,
                http_proxy=self.args["altclip_reward"].get("http_proxy", None),
                https_proxy=self.args["altclip_reward"].get("https_proxy", None)
            )
            self.altclip_reward_model.eval()
            self.altclip_reward_model.to(self.device)
            self.reward_models.append(self.altclip_reward_model)
            self.logger.info(f"AltCLIP reward model built successfully and on device: {self.device}")
        
        # Google Gemini OCR reward model: Use the remote server to compute the reward
        if self.args.get("google_gemini_ocr", False):
            gg_app_ids = self.args["google_gemini_ocr"]["app_ids"]
            gg_app_keys = self.args["google_gemini_ocr"]["app_keys"]
            if isinstance(gg_app_ids, list):
                assert (
                    len(gg_app_ids) == len(gg_app_keys)
                ), "The number of google_gemini_ocr app_ids and app_keys must be the same"
                num_app_ids = len(gg_app_ids)
                app_id_idx = self.rank % num_app_ids
                gg_app_id = gg_app_ids[app_id_idx]
                gg_app_key = gg_app_keys[app_id_idx]
            else:
                gg_app_id = gg_app_ids
                gg_app_key = gg_app_keys
            self.gg_text_ocr_reward_model = GG_OCR3(
                app_id=gg_app_id,
                app_key=gg_app_key,
                logger=self.logger,
                model_marker=self.args["google_gemini_ocr"].get("model_marker", "api_google_gemini-2.5-pro"),
            )
            self.reward_models.append(self.gg_text_ocr_reward_model)

        # Google Gemini Counting reward model: Use the remote server to compute the reward
        if self.args.get("google_gemini", False):
            google_gemini_app_ids = self.args["google_gemini"]["google_gemini_app_id"]
            google_gemini_app_keys = self.args["google_gemini"]["google_gemini_app_key"]
            if isinstance(google_gemini_app_ids, list):
                assert (
                    len(google_gemini_app_ids) == len(google_gemini_app_keys)
                ), "The number of google_gemini_app_ids and google_gemini_app_keys must be the same"
                num_app_ids = len(google_gemini_app_ids)
                app_id_idx = self.rank % num_app_ids
                google_gemini_app_id = google_gemini_app_ids[app_id_idx]
                google_gemini_app_key = google_gemini_app_keys[app_id_idx]
            else:
                google_gemini_app_id = google_gemini_app_ids
                google_gemini_app_key = google_gemini_app_keys
                
            self.gemini_reward_model = GoogleGeminiRewardModel(
                app_id=google_gemini_app_id,
                app_key=google_gemini_app_key,
                task_name=self.args["google_gemini"].get("google_gemini_task_name", "COUNT_CN_WITH_REASON"),
                model_marker=self.args["google_gemini"].get("model_marker", "api_google_gemini-2.5-pro"),
                logger=self.logger,
            )
            self.reward_models.append(self.gemini_reward_model)
        
        # HPS-v2 reward model: Build on local device since the model is small and the inference speed is fast
        if self.args.get("hps_clip_reward", False):
            self.hps_clip_reward_model = HPSClipRewardModel(
                device=self.device,
                clip_ckpt_path=self.args["hps_clip_reward"]["clip_ckpt_path"],
                hps_ckpt_path=self.args["hps_clip_reward"]["hps_ckpt_path"],
            )
            self.reward_models.append(self.hps_clip_reward_model)
        
        # Image-Reward model: Build on local device since the model is small and the inference speed is fast
        if self.args.get("image_reward", False):
            self.image_reward_model = ImageRewardModel(
                model_name=self.args["image_reward"]["ir_model_path"],
                device=self.device,
                med_config=self.args["image_reward"]["med_config"],
                http_proxy=self.args["image_reward"].get("ir_http_proxy", None),
                https_proxy=self.args["image_reward"].get("ir_https_proxy", None)
            )
            self.reward_models.append(self.image_reward_model)

        # Unified-Reward model: Use the remote server to compute the reward
        if self.args.get("unified_reward", False):
            ur_urls = self.args["unified_reward"]["ur_url"]
            if isinstance(ur_urls, list):
                num_urls = len(ur_urls)
                ur_url_idx = self.rank % num_urls
                ur_url = ur_urls[ur_url_idx]
                self.logger.info(f"Rank {self.rank} using unified-reward URL: {ur_url}")
            self.unified_reward_model = UnifiedRewardModel(
                api_url=ur_url,
                default_question_type=self.args["unified_reward"]["ur_default_question_type"],
                num_workers=self.args["unified_reward"]["ur_num_workers"],
            )
            self.reward_models.append(self.unified_reward_model)
        
        # PickScore reward model: Use the remote server to compute the reward
        if self.args.get("pick_score", False):
            self.pick_score_model = PickScoreRewardModel(
                device=self.device,
                http_proxy=self.args["pick_score"].get("ps_http_proxy", None),
                https_proxy=self.args["pick_score"].get("ps_https_proxy", None)
            )
            self.reward_models.append(self.pick_score_model)
        
        if self.args.get("hpsv3_reward", False):
            # url_port in the format of "url:port"
            hpsv3_url_port = self.args["hpsv3_reward"]["url_port"]
            if isinstance(hpsv3_url_port, list):
                num_url_ports = len(hpsv3_url_port)
                url_port_idx = self.rank % num_url_ports
                cur_url_port = hpsv3_url_port[url_port_idx]
            else:
                cur_url_port = hpsv3_url_port
            cur_hpsv3_url, cur_hpsv3_port = cur_url_port.split(":")
            self.hpsv3_reward_model = HPSV3RewardModel(
                url=cur_hpsv3_url,
                port=cur_hpsv3_port
            )
            self.reward_models.append(self.hpsv3_reward_model)

        ############################# TI2I reward models #############################
        # Google Gemini Editing reward model: Use the remote server to compute the reward
        if self.args.get("google_gemini_editing", False):
            google_gemini_app_ids = self.args["google_gemini_editing"]["google_gemini_editing_app_id"]
            google_gemini_app_keys = self.args["google_gemini_editing"]["google_gemini_editing_app_key"]
            if isinstance(google_gemini_app_ids, list):
                assert (
                    len(google_gemini_app_ids) == len(google_gemini_app_keys)
                ), "The number of google_gemini_app_ids and google_gemini_app_keys must be the same"
                num_app_ids = len(google_gemini_app_ids)
                app_id_idx = self.rank % num_app_ids
                google_gemini_app_id = google_gemini_app_ids[app_id_idx]
                google_gemini_app_key = google_gemini_app_keys[app_id_idx]
            else:
                google_gemini_app_id = google_gemini_app_ids
                google_gemini_app_key = google_gemini_app_keys
                
            self.gemini_editing_reward_model = GoogleGeminiRewardModel(
                app_id=google_gemini_app_id,
                app_key=google_gemini_app_key,
                task_name=self.args["google_gemini_editing"].get("google_gemini_editing_task_name", "Editing"),
                logger=self.logger,
                score_weights=self.args["google_gemini_editing"].get("google_gemini_editing_score_weights", None),
                model_marker=self.args["google_gemini_editing"].get("google_gemini_editing_model_marker", None),
            )
            self.reward_models.append(self.gemini_editing_reward_model)

        # Subject Driven reward model: Use the remote server to compute the reward
        if self.args.get("subject_driven_reward", False):
            google_gemini_app_ids = self.args["subject_driven_reward"]["google_gemini_subject_driven_app_id"]
            google_gemini_app_keys = self.args["subject_driven_reward"]["google_gemini_subject_driven_app_key"]
            if isinstance(google_gemini_app_ids, list):
                assert (
                    len(google_gemini_app_ids) == len(google_gemini_app_keys)
                ), "The number of google_gemini_app_ids and google_gemini_app_keys must be the same"
                num_app_ids = len(google_gemini_app_ids)
                app_id_idx = self.rank % num_app_ids
                google_gemini_app_id = google_gemini_app_ids[app_id_idx]
                google_gemini_app_key = google_gemini_app_keys[app_id_idx]
            else:
                google_gemini_app_id = google_gemini_app_ids
                google_gemini_app_key = google_gemini_app_keys
            
            if self.args["subject_driven_reward"].get("task_name", False):
                if self.args["subject_driven_reward"]["task_name"].get("consistency_face_weight", 0.0) > 0:
                    self.subject_driven_face_consistency_reward_model = SubDriCons_RM_Face(
                        http_proxy=self.args["subject_driven_reward"].get("face_http_proxy", None),
                        https_proxy=self.args["subject_driven_reward"].get("face_https_proxy", None),
                    )
                    self.reward_models.append(self.subject_driven_face_consistency_reward_model)

                if self.args["subject_driven_reward"]["task_name"].get("consistency_gemini_weight", 0.0) > 0:
                    self.subject_driven_consistency_reward_model = SubDriCons_RM_Gemini(
                        app_id=google_gemini_app_id,
                        app_key=google_gemini_app_key,
                        logger=self.logger,
                        model_marker=self.args["subject_driven_reward"].get("google_gemini_subject_driven_model_marker", None),
                    )
                    self.reward_models.append(self.subject_driven_consistency_reward_model)

                if self.args["subject_driven_reward"]["task_name"].get("semantic_alignment_weight", 0.0) > 0:
                    self.subject_driven_semantic_reward_model = SemAlign_RM(
                        app_id=google_gemini_app_id,
                        app_key=google_gemini_app_key,
                        logger=self.logger,
                        model_marker=self.args["subject_driven_reward"].get("google_gemini_subject_driven_model_marker", None),
                    )
                    self.reward_models.append(self.subject_driven_semantic_reward_model)

        # Google Gemini Editing reward model (based on different subtasks) 
        if self.args.get("editing_reward", False):
            google_gemini_app_ids = self.args["editing_reward"]["google_gemini_editing_app_id"]
            google_gemini_app_keys = self.args["editing_reward"]["google_gemini_editing_app_key"]
            if isinstance(google_gemini_app_ids, list):
                assert (
                    len(google_gemini_app_ids) == len(google_gemini_app_keys)
                ), "The number of google_gemini_app_ids and google_gemini_app_keys must be the same"
                num_app_ids = len(google_gemini_app_ids)
                app_id_idx = self.rank % num_app_ids
                google_gemini_app_id = google_gemini_app_ids[app_id_idx]
                google_gemini_app_key = google_gemini_app_keys[app_id_idx]
            else:
                google_gemini_app_id = google_gemini_app_ids
                google_gemini_app_key = google_gemini_app_keys
            if self.args["editing_reward"].get("weight", 0.0) > 0:
                self.editing_reward_model = EditingCons_RM_Gemini(
                    app_id=google_gemini_app_id,
                    app_key=google_gemini_app_key,
                    logger=self.logger,
                    model_marker=self.args["editing_reward"].get("google_gemini_editing_model_marker", None),
                )
                self.reward_models.append(self.editing_reward_model)
        
        # Editscore Reward Model
        if self.args.get("editscore", False):
            gg_app_id = self.args["editscore"]["gg_app_id"],
            gg_app_key = self.args["editscore"]["gg_app_key"],
            gg_api_version = self.args["editscore"]["gg_api_version"]
            gg_model_marker = self.args["editscore"]["gg_model_marker"]
            gg_max_retries = self.args["editscore"].get("gg_max_retries", 3)
            gg_timeout = self.args["editscore"].get("gg_timeout", 30)
            proxy_hosts = self.args["editscore"].get("proxy_hosts", [])
            proxy_port = self.args["editscore"].get("proxy_port", 8080)
            timeout = self.args["editscore"].get("timeout", 100)
            max_retries = self.args["editscore"].get("max_retries", 3)
            resize = self.args["editscore"].get("resize", 512)
            resize_mode = self.args["editscore"].get("resize_mode", "square")
            translate_before_eval = self.args["editscore"].get("translate_before_eval", True)

            if isinstance(proxy_hosts, list):
                num_proxy_hosts = len(proxy_hosts)
                proxy_host_idx = self.rank % num_proxy_hosts
                proxy_host = proxy_hosts[proxy_host_idx]
                self.editscore_reward_model = EditScoreRewardModel(
                    gg_app_id=gg_app_id,
                    gg_app_key=gg_app_key,
                    gg_api_version=gg_api_version,
                    gg_model_marker=gg_model_marker,
                    gg_max_retries=gg_max_retries,
                    gg_timeout=gg_timeout,
                    proxy_host=proxy_host,
                    proxy_port=proxy_port,
                    timeout=timeout,
                    max_retries=max_retries,
                    logger=self.logger,
                    resize=resize,
                    resize_mode=resize_mode,
                    translate_before_eval=translate_before_eval,
                )
            else:
                self.editscore_reward_model = EditScoreRewardModel(
                    gg_app_id=gg_app_id,
                    gg_app_key=gg_app_key,
                    gg_api_version=gg_api_version,
                    gg_model_marker=gg_model_marker,
                    gg_max_retries=gg_max_retries,
                    gg_timeout=gg_timeout,
                    proxy_host=proxy_hosts,
                    proxy_port=proxy_port,
                    timeout=timeout,
                    max_retries=max_retries,
                    logger=self.logger,
                    resize=resize,
                    resize_mode=resize_mode,
                    translate_before_eval=translate_before_eval,
                )
            self.reward_models.append(self.editscore_reward_model)
  
        # Edit Reward Model
        if self.args.get("edit_ref_rm", False):
            edit_ref_url_port = self.args["edit_ref_rm"]["edit_ref_url_port"]
            assert isinstance(edit_ref_url_port, list), f"edit_ref_url_port must be a list, but got {type(edit_ref_url_port)}"

            num_proxy_hosts = len(edit_ref_url_port)
            proxy_host_idx = self.rank % num_proxy_hosts
            proxy_host = edit_ref_url_port[proxy_host_idx]
            self.edit_ref_rm = EditRefRM(proxy_host)
            self.reward_models.append(self.edit_ref_rm)

        ############################# Reward Models Setting #############################
        # Qwen OCR reward model settings
        if self.args.get("text_ocr", False):
            ocr_args = self.args["text_ocr"]
            # If url is a list, assign one url for each rank uniformly
            url = ocr_args.get("ocr_url")
            if isinstance(url, list):
                num_urls = len(url)
                url_idx = self.rank % num_urls
                url = url[url_idx]
                self.logger.info(f"Rank {self.rank} using ocr-server URL: {url}")
                
            self.ocr_reward_kwargs = {
                "url": url,
                "max_workers": ocr_args.get("ocr_num_workers", 8),
                "show_progress": True,
                "if_split_by_character": ocr_args.get("if_split_by_character"),
                "reward_metric": ocr_args.get("ocr_reward_metric")
            }
        
        ############################ Preference Reward Model Server ############################
        if self.args.get("yingyong_preference_reward", False):
            # url_port in the format of "url:port"
            preference_rw_url_port = self.args["yingyong_preference_reward"]["preference_rw_url_port"]
            if isinstance(preference_rw_url_port, list):
                num_url_ports = len(preference_rw_url_port)
                url_port_idx = self.rank % num_url_ports
                cur_url_port = preference_rw_url_port[url_port_idx]
            else:
                cur_url_port = preference_rw_url_port
            cur_preference_rw_url, cur_preference_rw_port = cur_url_port.split(":")
            self.preference_reward_model = PreferenceRewardModel(
                url=cur_preference_rw_url,
                port=cur_preference_rw_port
            )
            self.reward_models.append(self.preference_reward_model)
        
        # Initialize reward model weights only for activated models
        # Initialize all possible reward loss keys
        self.reward_weights = {}
        self.all_reward_loss_keys = []
        for model in self.reward_models:
            model_name = type(model).__name__
            self.all_reward_loss_keys.append(f"{model_name}_dummyloss")
            if model_name == 'TextOCRGroundingQwenVL':
                weight = self.args.get('text_ocr_weight', 1.0)
            elif model_name == 'GG_OCR3':
                weight = self.args.get('google_gemini_ocr_weight', 1.0)
            elif model_name == 'GoogleGeminiRewardModel':
                weight = self.args.get('google_gemini_weight', 1.0)
            elif model_name == 'HPSClipRewardModel':
                weight = self.args.get('hps_clip_weight', 1.0)
            elif model_name == 'ImageRewardModel':
                weight = self.args.get('image_reward_weight', 1.0)
            elif model_name == 'UnifiedRewardModel':
                weight = self.args.get('unified_reward_weight', 1.0)
            elif model_name == 'PickScoreRewardModel':
                weight = self.args.get('pick_score_weight', 1.0)
            elif model_name == 'HPSV3RewardModel':
                weight = self.args.get('hpsv3_reward_weight', 0.1)
            elif model_name == 'PreferenceRewardModel':
                weight = self.args.get('preference_reward_weight', 1.0)
            elif model_name == 'AltCLIPRM':
                weight = self.args.get('altclip_reward_weight', 1.0)
            elif model_name == 'SubDriCons_RM_Face':
                weight = self.args["subject_driven_reward"]["task_name"].get("consistency_face_weight", 1.0)
            elif model_name == 'SubDriCons_RM_Gemini':
                weight = self.args["subject_driven_reward"]["task_name"].get("consistency_gemini_weight", 1.0)
            elif model_name == 'SubDriSem_RM':
                weight = self.args["subject_driven_reward"]["task_name"].get("semantic_alignment_weight", 1.0)
            elif model_name == 'EditingCons_RM_Gemini':
                weight = self.args["editing_reward"].get("weight", 1.0)
            elif model_name == 'EditScoreRewardModel':
                weight = self.args.get('editscore_weight', 1.0)
            elif model_name == 'EditRefRM':
                weight = self.args.get('edit_ref_rm_weight', 1.0)
            else:
                weight = 1.0
            self.reward_weights[model_name] = weight

        # Normalize weights
        total_weight = sum(self.reward_weights.values())
        if total_weight > 0:
            self.reward_weights = {k: v/total_weight for k, v in self.reward_weights.items()}
        else:
            self.logger.warning("No reward models activated or all weights are 0!")
            self.reward_weights = {type(model).__name__: 1.0/len(self.reward_models) for model in self.reward_models}

        # t2i and ti2i may use difference reward systems
        rm_dict = {type(rm).__name__:rm for rm in self.reward_models}

        self.t2i_reward_models = [rm_dict[rm] for rm in self.args.get("t2i_reward_models", [])]
        self.t2i_reward_weights = {rm: self.reward_weights[rm] for rm in self.args.get("t2i_reward_models", [])}
        self.ti2i_reward_models = [rm_dict[rm] for rm in self.args.get("ti2i_reward_models", [])]
        self.ti2i_reward_weights = {rm: self.reward_weights[rm] for rm in self.args.get("ti2i_reward_models", [])}

    def task_init(self, args, all_dataset_keys=None):
        self.sampling_probs_dict = json.loads(args.sampling_probs)
        self.all_dataset_keys = sorted(list(self.sampling_probs_dict.keys()))

        # Define what dummy token are incurred by each task.
        self.dummy_to_tasks = dict(
            t2i={"t2i", "editing", "subject_driven", "interleave", "face_id_clip"},
            mmu={"mmu", "mmu_interleave", "face_id_clip"},
            face={},
        )

        # Define the sequence batch size for each task for long sequence training.
        self.seq_batch_size = args.get("seq_batch_size", {})
        if isinstance(self.seq_batch_size, str):
            self.seq_batch_size = json.loads(self.seq_batch_size)

        ############################# GRPO Core Parameters #############################
        # Training strategy
        self.training_strategy = args.get("training_strategy", "all")  # "all", "progressive", "random", "decay", "dynamic"
        assert self.training_strategy in ["all", "progressive", "random", "decay", "dynamic"], f"Invalid training strategy: {self.training_strategy}"
        self.same_x0 = args.get("same_x0", False)
        self.use_extra_low_timesteps_kl = args.get("use_extra_low_timesteps_kl", False)
        self.multi_reward_mix = args.get("multi_reward_mix", "reward_aggr")
        assert self.multi_reward_mix in ["reward_aggr", "reward_mix"], f"Invalid multi-reward mix: {self.multi_reward_mix}"

        # Training objective
        self.training_obj = args.get("training_obj", "advantage")  # "advantage", "reward"
        assert self.training_obj in ["advantage"], f"Invalid training objective: {self.training_obj}"

        # advantage normalization strategy
        self.advantage_norm = args.get("advantage_norm", "mean_std")
        assert self.advantage_norm in ["mean_std", "trimmed_mean"], f"Invalid advantage normalization strategy: {self.advantage_norm}"
        if self.advantage_norm == "trimmed_mean":
            self.trimmed_ratio = args.get("trimmed_ratio", 0.25)

        # Loss and reward parameters
        self.kl_weight = args.get("kl_weight")
        self.adv_clip_max = args.get("adv_clip_max")
        self.clip_range = args.get("clip_range")
        self.quant_reward = args.get("quant_reward", False)
        
        # Timesteps configuration
        self.num_timesteps = int(args.get("num_train_timesteps"))
        self.num_train_timesteps = int(self.num_timesteps * args.get("timestep_fraction"))
        args.diff_infer_steps = self.num_timesteps  # Make GRPO training timesteps consistent with sampling timesteps
        
        # GRPO iterations and generations
        self.num_grpo_iterations = int(args.get("num_grpo_iterations"))
        assert self.num_grpo_iterations == 1, f"Invalid value of num_grpo_iterations ({self.num_grpo_iterations}), which should be 1."
        self.num_generations = int(args.get("num_generations"))

        # Pipeline configuration
        self.pipeline_name = "transfusion_with_logprob"

        ############################# GRPO Training Strategy Parameters #############################
        self.timesteps_group_size = args.get("timesteps_group_size")
        self.train_iters_per_timesteps_group = args.get("train_iters_per_timesteps_group")
        self.timesteps_group_overlap = args.get("timesteps_group_overlap", False)
        self.mixgrpo_stride = args.get("mixgrpo_stride", 1)

        if self.training_strategy == "decay":
            self.decay_kwargs = {
                "max_iters_per_group": args.get("max_iters_per_group", 100),
                "min_iters_per_group": args.get("min_iters_per_group", 25),
            }
        elif self.training_strategy == "dynamic":
            self.dynamic_kwargs = {
                "dynamic_t1": args.get("dynamic_t1", 12),
                "dynamic_k": args.get("dynamic_k", 0.5),
                "dynamic_y0": args.get("dynamic_y0", 5)
            }

        ############################# Memory and Visualization Settings #############################
        # Memory optimization
        self.ref_model_cpu_offload = args.get("model_cpu_offload", False)
        self.reference_model_on_cpu = False
        
        # Visualization settings
        self.visualize_reward = args.get("visualize_reward", False)
        self.visualize_every = args.get("visualize_every", 100)

        # Gradient accumulation
        # We always accumulate gradients across timesteps; we want gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        if self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
            if self.use_extra_low_timesteps_kl:
                args.gradient_accumulation_steps = int((self.timesteps_group_size+1) * args.gradient_accumulation_steps)
            else:
                args.gradient_accumulation_steps = int(self.timesteps_group_size * args.gradient_accumulation_steps)
        else:
            args.gradient_accumulation_steps = int(self.num_train_timesteps * args.gradient_accumulation_steps)

    def build_reference_model(self):
        """
        Build the reference model. Note that model must be in `.eval()` mode.
        """
        self.logger.info("Building reference model...")
        factor_kwargs = {"device": "cpu", "dtype": PRECISION_TO_TYPE[self.args.precision]}
        assert self.args.get('pretrained_reference_model_ckpt', None) is not None, "pretrained_reference_model_ckpt is not set."
        self.ref_model, _ = build_model(self.args, self.args.pretrained_reference_model_ckpt, logger=self.logger, **factor_kwargs)
        self.ref_model.requires_grad = False
        self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()
        self.logger.info(f"Reference model built on device: {self.device}")

        set_manual_seed(self.args.global_seed + self.dp_rank)
    
    def extract_reward_types_from_tags(self, reward_tags):
        """Extract reward types from reward tags."""
        if reward_tags is None:
            return []
            
        rewards = []
        if "count" in reward_tags:
            rewards.append("GoogleGeminiRewardModel")
        if "ocr" in reward_tags:
            # TODO: 目前也用gemini来看ocr准确度
            # rewards.append("TextOCRGroundingQwenVL")
            rewards.append("GoogleGeminiRewardModel")
        if "quality" in reward_tags:
            rewards.extend(["ImageRewardModel", "UnifiedRewardModel", "PickScoreRewardModel", "HPSClipRewardModel", "HPSV3RewardModel"])

        # Check if all rewards are available
        available_rewards = []
        for reward in rewards:
            if reward in self.reward_weights:
                available_rewards.append(reward)
            else:
                self.logger.warning(f"Reward model {reward} is not available, so it is removed from the list of rewards.")
        
        # If no reward models are available, use ImageRewardModel as a fallback
        if available_rewards == []:
            available_rewards = ["ImageRewardModel"]
        
        return available_rewards

    def compute_reward(self, images, input_prompts, reward_tags=None, src_images=None, use_face_rewards=None, sem_points=None, subtask=None, ref_images=None):
        assert (
            len(images) == len(input_prompts)
        ), f"length of `images` ({len(images)}) must be equal to length of `input_prompts` ({len(input_prompts)})"
        self.logger.info(f"reward_tags: {reward_tags}")
        
        assert len(input_prompts) == 1, f"we support batch=1 only, but given batch={len(input_prompts)}"
        # Initialize results
        rewards_dict = {}
        successes_dict = {}
        if reward_tags is None:
            cur_reward_models = self.reward_models
            cur_reward_weights = self.reward_weights
        elif reward_tags[0] == 't2i':
            cur_reward_models = self.t2i_reward_models
            cur_reward_weights = self.t2i_reward_weights
        else:
            assert reward_tags[0] == 'ti2i'
            cur_reward_models = self.ti2i_reward_models
            cur_reward_weights = self.ti2i_reward_weights

        # Determine which reward models to use
        # if reward_tags is not None:
        #     reward_types = self.extract_reward_types_from_tags(reward_tags)
        #     # Filter reward models
        #     cur_reward_models = [model for model in self.reward_models if type(model).__name__ in reward_types]
        #     cur_reward_weights = {model_name: self.reward_weights[model_name] for model_name in reward_types}
        # else:
        #     cur_reward_models = self.reward_models
        #     cur_reward_weights = self.reward_weights
        
        self.logger.info(f"Using reward models: {[type(model).__name__ for model in cur_reward_models]}")
        
        # Create a thread pool for parallel reward computation
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cur_reward_models)) as executor:
            # Submit all reward computation tasks
            future_to_model = {
                executor.submit(self._compute_single_reward, reward_model, images, input_prompts, src_images, use_face_rewards, sem_points, None, ref_images): reward_model 
                for reward_model in cur_reward_models
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                reward_model = future_to_model[future]
                model_name = type(reward_model).__name__
                try:
                    model_rewards, model_successes = future.result()
                    # self.logger.info(f'lucas debug {model_name=}: {model_rewards=}, {model_successes=}')
                    rewards_dict[model_name] = model_rewards
                    successes_dict[model_name] = model_successes
                except Exception as e:
                    self.logger.info(f"Error computing reward with {model_name}: {e}")
                    rewards_dict[model_name] = [0.0] * len(input_prompts)
                    successes_dict[model_name] = [0] * len(input_prompts)
                    continue

        # Merge rewards based on weights
        merged_rewards = [0.0] * len(input_prompts)
        merged_successes = [0] * len(input_prompts)
        
        # First check if all models are successful for each sample
        for i in range(len(merged_rewards)):
            all_success = True
            for model_name in cur_reward_weights.keys():
                if model_name in successes_dict and successes_dict[model_name][i] != 1:
                    all_success = False
                    break
            
            if all_success:
                # Only compute weighted sum if all models are successful
                for model_name, weight in cur_reward_weights.items():
                    if model_name in rewards_dict:
                        merged_rewards[i] += rewards_dict[model_name][i] * weight
                merged_successes[i] = 1
        # self.logger.info(f'lucas debug: {merged_rewards=}, {merged_successes=}, {rewards_dict=}, {successes_dict=}')
        return merged_rewards, merged_successes, rewards_dict, successes_dict

    def _compute_single_reward(self, reward_model, images, input_prompts, 
                               src_images=None, 
                               use_face_rewards=False, 
                               sem_points=None,
                               subtask=None,
                               ref_images=None):
        """Compute reward for a single reward model."""
        reward_model_name = type(reward_model).__name__
        try:
            if reward_model_name == 'TextOCRGroundingQwenVL':
                gt = [reward_model.parse_gt(text) for text in input_prompts]
                eval_res = reward_model.eval(
                    images,
                    gt,
                    **self.ocr_reward_kwargs,
                )
                rewards = [eval_res[i][self.ocr_reward_kwargs["reward_metric"]] for i in range(len(eval_res))]
                if self.quant_reward:
                    def _quant_reward_fn(reward):
                        if reward < 0.97:
                            return 0.0
                        else:
                            return 1.0
                    rewards = [_quant_reward_fn(reward) for reward in rewards]
                successes = [1] * len(rewards)

            elif reward_model_name == 'GG_OCR3':
                rewards, successes = reward_model.eval(
                    images,
                    input_prompts,
                    max_workers=1,
                    show_progress=True,
                    gemini_try_times=self.args["google_gemini_ocr"].get("gemini_try_times", 2),
                    timeout=self.args["google_gemini_ocr"].get("timeout", 30),
                )
            
            elif reward_model_name == 'GoogleGeminiRewardModel':
                rewards, successes = reward_model.eval(
                    images,
                    input_prompts,
                    src_images,
                    max_workers=1,
                    show_progress=True,
                    gemini_try_times=self.args["google_gemini"].get("gemini_try_times", 2),
                    timeout=self.args["google_gemini"].get("timeout", 30),
                )

            elif reward_model_name == 'HPSClipRewardModel':
                rewards = reward_model(images, input_prompts)
                successes = [1] * len(rewards)
            
            elif reward_model_name == 'AltCLIPRM':
                rewards = reward_model(images, input_prompts)
                successes = [1] * len(rewards)
            
            elif reward_model_name == 'ImageRewardModel':
                rewards = reward_model(images, input_prompts)
                successes = [1] * len(rewards)

            elif reward_model_name == 'UnifiedRewardModel':
                rewards, successes_bool = reward_model(images, input_prompts)
                rewards = [float(reward) if success else 0.0 for reward, success in zip(rewards, successes_bool)]
                successes = [1 if success else 0 for success in successes_bool]

            elif reward_model_name == 'PickScoreRewardModel':
                rewards = reward_model(images, input_prompts)
                successes = [1] * len(rewards)
            
            elif reward_model_name == 'HPSV3RewardModel':
                rewards, successes = reward_model(images, input_prompts)
            
            elif reward_model_name == 'PreferenceRewardModel':
                rewards, successes = reward_model(images, input_prompts)
            
            elif reward_model_name == 'SubDriCons_RM_Gemini':
                rewards, successes = reward_model.eval(
                    images,
                    input_prompts,
                    src_images,
                    use_face_rewards=use_face_rewards,
                    max_workers=1,
                    show_progress=False,
                    gemini_try_times=self.args.get("gemini_try_times", 30),
                    timeout=self.args.get("gemini_timeout", 3600)
                )
            
            elif reward_model_name == 'SemAlign_RM':
                rewards, successes = reward_model.eval(
                    images=images,
                    prompts=sem_points,
                    max_workers=1,
                    show_progress=False,
                    gemini_try_times=self.args.get("gemini_try_times", 30),
                    timeout=self.args.get("gemini_timeout", 3600),
                )

            elif reward_model_name == 'EditingCons_RM_Gemini':
                rewards, successes = reward_model.eval(
                    images=images,
                    prompts=input_prompts,
                    src_images=src_images,
                    max_workers=1,
                    show_progress=False,
                    resize_images=False, # TODO: Debug
                    subtask=subtask,
                    gemini_try_times=self.args.get("gemini_try_times", 30),
                    timeout=self.args.get("gemini_timeout", 3600),
                )

            elif reward_model_name == 'EditScoreRewardModel':
                rewards, successes = reward_model(
                    input_images=src_images,
                    output_images=images,
                    texts=input_prompts,
                    rank=self.rank
                )
            elif reward_model_name == 'EditRefRM':
                rewards, successes = reward_model(
                    src_images=src_images,
                    prompts=input_prompts,
                    ref_images=ref_images,
                    gen_images=images,
                )
            else:
                raise ValueError(f"Unknown reward model: {reward_model_name}")

            # Verify the length of results matches input
            assert len(rewards) == len(input_prompts), \
                f"Length mismatch in {reward_model_name}: rewards ({len(rewards)}) != input_prompts ({len(input_prompts)})"
            assert len(successes) == len(input_prompts), \
                f"Length mismatch in {reward_model_name}: successes ({len(successes)}) != input_prompts ({len(input_prompts)})"

            # Visualize images and rewards if needed
            if self.visualize_reward and self.ss.update_steps % self.visualize_every == 0:
                visualize_dir = os.path.join(self.exp_dir, "reward_visualizations")
                os.makedirs(visualize_dir, exist_ok=True)
                if src_images is not None:
                    plt.figure(figsize=(16, 8))
                    
                    # 左图：src_images
                    plt.subplot(1, 2, 1)
                    plt.imshow(src_images[0])
                    plt.title("Source Image")
                    plt.axis('off')
                    
                    # 右图：images0
                    plt.subplot(1, 2, 2)
                    plt.imshow(images[0])
                    if input_prompts[0] is not None:
                        plt.title(f"Reward Score: {rewards[0]:.4f}, Instruction: {input_prompts[0][:20]}")
                    else:
                        plt.title(f"Reward Score: {rewards[0]:.4f}")
                    plt.axis('off')
                else:
                    # 只有一个图像
                    plt.figure(figsize=(8, 8))
                    plt.imshow(images[0])
                    if input_prompts[0] is not None:
                        plt.title(f"Reward Score: {rewards[0]:.4f}, Instruction: {input_prompts[0][:20]}")
                    else:
                        plt.title(f"Reward Score: {rewards[0]:.4f}")
                    plt.axis('off')
                
                save_path = os.path.join(
                    visualize_dir,
                    f"reward_{reward_model_name}_step_{self.ss.update_steps}_sample_{self.rank}_reward_{rewards[0]:.4f}.png"
                )
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                outf = os.path.join(visualize_dir, f"reward_rank{self.rank}.jsonl")
                with open(outf, 'a') as fd:
                    item = {
                        'prompt': input_prompts[0],
                        'reward_{reward_model_name}': rewards[0],
                        'rank': self.rank,
                        'step': self.ss.update_steps,
                        'image': save_path,
                    }
                    fd.write(json.dumps(item, ensure_ascii=False) + '\n')

            return rewards, successes

        except Exception as e:
            self.logger.info(f"Error in _compute_single_reward-{reward_model_name}: {e}")
            return [0.0] * len(input_prompts), [0] * len(input_prompts)
    
    def compute_advantages(
            self,
            gathered_rewards: torch.Tensor,
            reward_mask: torch.Tensor,
            num_generations: int,
            process_slice: slice,
            reward_tags=None,
            compute_exploration_score=False,
    ):
        """
        Compute advantages using different normalization strategies across the valid rewards.
        
        Args:
            gathered_rewards (torch.Tensor): Gathered rewards from all processes
            reward_mask (torch.Tensor): Mask for valid rewards
            num_generations (int): Number of generations per prompt
            process_slice (slice): Slice to keep only the local part of the data
            
        Returns:
            torch.Tensor: Normalized advantages for the current process
        """
        # Reshape rewards and successes to (num_prompts, num_generations)
        if self.multi_reward_mix == "advantage_aggr":
            rewards_reshaped = {}
            mean_grouped_rewards = {}
            std_grouped_rewards = {}
            for reward_name, gathered_reward in gathered_rewards.items():
                rewards_reshaped[reward_name] = gathered_reward.view(-1, num_generations)
                mean_grouped_rewards[reward_name] = torch.zeros_like(rewards_reshaped[reward_name][:, 0])
                std_grouped_rewards[reward_name] = torch.zeros_like(rewards_reshaped[reward_name][:, 0])
            if compute_exploration_score:
                avg_exploration_score, avg_range, avg_std = average_exploration_score(
                    [rewards_reshaped[reward_name].cpu().numpy().tolist() for reward_name in rewards_reshaped.keys()]
                )
                self.logger.info(f"avg_exploration_score: {avg_exploration_score}, avg_range: {avg_range}, avg_std: {avg_std}")

        elif self.multi_reward_mix == "reward_aggr":
            rewards_reshaped = gathered_rewards.view(-1, num_generations)
            mean_grouped_rewards = torch.zeros_like(rewards_reshaped[:, 0])
            std_grouped_rewards = torch.zeros_like(rewards_reshaped[:, 0])
            if compute_exploration_score:
                avg_exploration_score, avg_range, avg_std = average_exploration_score(rewards_reshaped.cpu().numpy().tolist())
                self.logger.info(f"avg_exploration_score: {avg_exploration_score}, avg_range: {avg_range}, avg_std: {avg_std}")

        # Determine which reward models to use
        if reward_tags is not None:
            reward_types = self.extract_reward_types_from_tags(reward_tags)
            # Filter reward models
            cur_reward_models = [model for model in self.reward_models if type(model).__name__ in reward_types]
            cur_reward_weights = {model_name: self.reward_weights[model_name] for model_name in reward_types}
        else:
            cur_reward_models = self.reward_models
            cur_reward_weights = self.reward_weights
        
        if self.advantage_norm == "mean_std":
            # Compute mean and std only for valid rewards
            # For each prompt, compute mean/std only over valid generations
            if self.multi_reward_mix == "advantage_aggr":
                for reward_name, rewards in rewards_reshaped.items():
                    for i in range(rewards.size(0)):
                        valid_rewards = rewards[i][reward_mask[i]]
                        if len(valid_rewards) > 0:
                            mean_grouped_rewards[reward_name][i] = valid_rewards.mean()
                            std_grouped_rewards[reward_name][i] = valid_rewards.std()
                        else:
                            # If no valid rewards, use the original reward
                            mean_grouped_rewards[reward_name][i] = rewards[i].mean()
                            std_grouped_rewards[reward_name][i] = rewards[i].std()
            elif self.multi_reward_mix == "reward_aggr":
                for i in range(rewards_reshaped.size(0)):
                    valid_rewards = rewards_reshaped[i][reward_mask[i]]
                    if len(valid_rewards) > 0:
                        mean_grouped_rewards[i] = valid_rewards.mean()
                        std_grouped_rewards[i] = valid_rewards.std()
                    else:
                        # If no valid rewards, use the original reward
                        mean_grouped_rewards[i] = rewards_reshaped[i].mean()
                        std_grouped_rewards[i] = rewards_reshaped[i].std()
                        
        elif self.advantage_norm == "trimmed_mean":
            # For trimmed mean, first filter valid rewards, then sort
            if self.multi_reward_mix == "advantage_aggr":
                for reward_name, rewards in rewards_reshaped.items():
                    for i in range(rewards.size(0)):
                        valid_rewards = rewards[i][reward_mask[i]]
                        if len(valid_rewards) > 0:
                            # Sort valid rewards
                            sorted_rewards = valid_rewards.sort()[0]
                            len_sorted_rewards = len(sorted_rewards)
                            trim_size = min(int(len_sorted_rewards * self.trimmed_ratio), len_sorted_rewards - 1)
                            trimmed_rewards = sorted_rewards[trim_size:]
                            
                            mean_grouped_rewards[reward_name][i] = trimmed_rewards.mean()
                            std_grouped_rewards[reward_name][i] = trimmed_rewards.std()
                        else:
                            # If no valid rewards, use the original reward
                            mean_grouped_rewards[reward_name][i] = rewards[i].mean()
                            std_grouped_rewards[reward_name][i] = rewards[i].std()
                            
            elif self.multi_reward_mix == "reward_aggr":
                for i in range(rewards_reshaped.size(0)):
                    valid_rewards = rewards_reshaped[i][reward_mask[i]]
                    if len(valid_rewards) > 0:
                        # Sort valid rewards
                        sorted_rewards = valid_rewards.sort()[0]
                        len_sorted_rewards = len(sorted_rewards)
                        trim_size = min(int(len_sorted_rewards * self.trimmed_ratio), len_sorted_rewards - 1)
                        trimmed_rewards = sorted_rewards[trim_size:]
                        
                        mean_grouped_rewards[i] = trimmed_rewards.mean()
                        std_grouped_rewards[i] = trimmed_rewards.std()
                    else:
                        # If no valid rewards, use the original reward
                        mean_grouped_rewards[i] = rewards_reshaped[i].mean()
                        std_grouped_rewards[i] = rewards_reshaped[i].std()

        if self.multi_reward_mix == "advantage_aggr":
            # Normalize the rewards to compute the advantages for each reward model
            advantages = {}
            advantages_aggr = torch.zeros_like(next(iter(gathered_rewards.values()))[process_slice])
            for reward_name, rewards in gathered_rewards.items():
                mean_rewards = mean_grouped_rewards[reward_name].repeat_interleave(num_generations, dim=0)
                std_rewards = std_grouped_rewards[reward_name].repeat_interleave(num_generations, dim=0)
                advantages[reward_name] = (rewards - mean_rewards) / (std_rewards + 1e-4)

                # Slice to keep only the local part of the data
                advantages[reward_name] = advantages[reward_name][process_slice]
            
            # weight
            for reward_name, weight in cur_reward_weights.items():
                if reward_name in advantages:
                    advantages_aggr += weight * advantages[reward_name]

            if compute_exploration_score:
                return advantages_aggr, avg_exploration_score
            return advantages_aggr

        elif self.multi_reward_mix == "reward_aggr":
            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
            advantages = (gathered_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
          
            # Slice to keep only the local part of the data
            advantages = advantages[process_slice]

            if compute_exploration_score:
                return advantages, avg_exploration_score
            return advantages
    
    def gen_reward_mask(self, successes):
        """
        Generate a mask for valid rewards.
        """
        successes_reshaped = successes.view(-1, self.num_generations)
        reward_mask = successes_reshaped.bool()
        return reward_mask
    
    def prepare_model_grpo_t2i_inputs(self, batch: Dict, device: Union[int, str], samples: Optional[List[Dict]]=None, timesteps_train: List[int]=None, **kwargs):
        input_prompts = batch["text"]
        batch_size = len(input_prompts)
        n_tokens = sum([len(prompt) for prompt in input_prompts])
        # TODO: 目前只支持每个rank bs=1的情况
        if "reward_tags" in batch:
            reward_tags = batch["reward_tags"][0]
        else:
            reward_tags = None

        t_start = time.time()
        ################################ 1. sample images (deepspeed) ################################
        self.model_engine.module.eval()
        with torch.inference_mode():
            with unwrap_model_for_generation_deepspeed(self.model_engine) as unwrapped_model:
                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    self.sampler.pipeline.model = unwrapped_model
                    
                    if self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
                        # Set seed for each prompt, so as to ensure that generating samples using the same initial latents;
                        # Set `deterministic` to False for the timesteps that are trained, otherwise set to True;
                        # In this way, different samples are generated for the same prompt by performing SDE operations at trainable 
                        # intermediate timesteps. (The variance in sde is different for each rank due to the initial different seeds.)
                        seeds = batch["seeds"]
                        determistic = [True] * self.num_train_timesteps
                        for timestep_i in timesteps_train:
                            determistic[timestep_i] = False
                    else:
                        # seeds = None
                        # determistic = False
                        seeds = batch["seeds"]
                        determistic = [False] * self.num_train_timesteps

                    out_dict = self.sampler.batch_x2image(
                        [input_prompts],
                        seed=seeds,
                        verbose=1,
                        task="t2i",
                        sequence_template=self.args.sequence_template,
                        predict_image_shape_token=self.args.predict_image_shape_token,
                        sample_image_size=self.args.sample_image_size,
                        return_only_samples=False,
                        pipeline_kwargs={"kl_weight": self.kl_weight, "determistic": determistic},
                    )
        t_rollout = time.time()
        self.model_engine.module.train()

        gen_imgs = out_dict["samples"]["samples"]  # list of PIL.Image.Image
        all_latents, all_log_probs, all_ref_prev_latents_mean, model_input_extra_kwargs = out_dict["extra_outputs"]
        # Convert inference tensors to normal tensors that can be used in autograd, we need .clone() to create new tensors from inference tensors
        all_latents = [latent.clone().detach() for latent in all_latents]
        all_log_probs = [log_prob.clone().detach() for log_prob in all_log_probs]
        all_ref_prev_latents_mean = [ref_prev_latents_mean.clone().detach() for ref_prev_latents_mean in all_ref_prev_latents_mean]
        model_input_extra_kwargs = {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in model_input_extra_kwargs.items()}
        model_input_extra_kwargs["return_loss"] = False

        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps)
        if self.kl_weight > 0:
            all_ref_prev_latents_mean = torch.stack(all_ref_prev_latents_mean, dim=1)  # (batch_size, num_steps, ...)
        else:
            all_ref_prev_latents_mean = None
        timesteps = self.sampler.pipeline.scheduler.timesteps.repeat(
            len(input_prompts), 1
        )  # (batch_size, num_steps)

        ################################ 2. compute rewards ################################
        rewards, successes, rewards_dict, successes_dict = self.compute_reward(gen_imgs, input_prompts, reward_tags)
        t_reward = time.time()
        self.logger.info(f'prepare input time cost: total={t_reward-t_start:.2f}s, rollout={t_rollout-t_start:.2f}s, reward={t_reward-t_rollout:.2f}s')

        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        successes = torch.from_numpy(np.array(successes)).int().to(self.device)
        rewards_dict = {k: torch.from_numpy(np.array(v)).float() for k, v in rewards_dict.items()}
        rewards_dict = dict(sorted(rewards_dict.items()))
        successes_dict = {k: torch.from_numpy(np.array(v)).int() for k, v in successes_dict.items()}

        if samples is None:
            samples = []
        samples.append(
            {
                "prompts": input_prompts,
                "timesteps": timesteps,
                "latents": all_latents[:, :-1], # each entry is the latent before timestep t
                "next_latents": all_latents[:, 1:], # each entry is the latent after timestep t
                "log_probs": all_log_probs,
                "ref_prev_latents_mean": all_ref_prev_latents_mean,
                "rewards": rewards,
                "successes": successes,
                "model_input_extra_kwargs": model_input_extra_kwargs,
                "rewards_dict": rewards_dict,
                "successes_dict": successes_dict,
            }
        )

        if self.multi_reward_mix == "advantage_aggr":
            gathered_rewards = {}
            for model_name, model_rewards in rewards_dict.items():
                gathered_rewards[model_name] = gather_tensor(model_rewards.to(self.device)).view(-1) # [world_size, batch_size]
                dist.barrier()
        elif self.multi_reward_mix == "reward_aggr":
            gathered_rewards = gather_tensor(samples[0]["rewards"]).view(-1) # [world_size, batch_size]
            dist.barrier()

        ################################ 3. compute advantages ################################
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        gathered_successes = gather_tensor(samples[0]["successes"]).view(-1)  # [world_size, batch_size]
        reward_mask = self.gen_reward_mask(gathered_successes)
        self.logger.info(f"gathered_rewards: {gathered_rewards}")
        self.logger.info(f"gathered_successes: {gathered_successes.bool()}")

        if self.training_obj == "advantage":
            # Get process slice for local data
            process_slice = slice(
                dist.get_rank() * len(samples[0]["prompts"]),
                (dist.get_rank() + 1) * len(samples[0]["prompts"]),
            )
            # Compute advantages using the specified normalization strategy
            advantages = self.compute_advantages(
                gathered_rewards,
                reward_mask,
                self.num_generations,
                process_slice,
                reward_tags,
            )
            samples[0]["advantages"] = advantages

        return samples, batch_size, n_tokens

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], samples: Optional[List[Dict]]=None, timesteps_train: List[int]=None, **kwargs):
        if batch["dtype"][0] == "t2i":
            inputs = self.prepare_model_grpo_t2i_inputs(batch, device, samples, timesteps_train, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs

    def train_step(
            self,
            batch,
            timestep_i: int,
            cache_samples=None,
            cur_batch_size=None,
            n_tokens=None,
            timesteps_train: List[int]=None,
            only_kl_loss: bool=False,
    ):
        start1 = time.time()
        if cache_samples is None:
            samples, cur_batch_size, n_tokens = self.prepare_model_inputs(batch, self.device, timesteps_train=timesteps_train)
        else:
            samples = cache_samples
            assert (
                cur_batch_size is not None and n_tokens is not None
            ), "`cur_batch_size` and `n_tokens` must be provided if `cache_samples` is not None"
        torch.cuda.synchronize()
        duration1 = time.time() - start1

        start2 = time.time()
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                self.model_engine,
                self.sampler.pipeline,
                samples[0],
                timestep_i,
                samples[0]["model_input_extra_kwargs"],
            )
        if self.kl_weight > 0:
            prev_sample_mean_ref = samples[0]["ref_prev_latents_mean"][:, timestep_i]
        else:
            prev_sample_mean_ref = None
        
            # if self.kl_weight > 0:
                # if self.model_cpu_offload and self.reference_model_on_cpu:
                #     self.reference_model_on_cpu = False
                #     self.ref_model = move_model_params_and_grads_to(self.ref_model, self.device)
                # with torch.inference_mode():
                #     prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref = compute_log_prob(
                #         self.ref_model,
                #         self.sampler.pipeline,
                #         samples[0],
                #         timestep_i,
                #         samples[0]["model_input_extra_kwargs"],
                #     )
                # if self.model_cpu_offload:
                #     self.reference_model_on_cpu = True
                #     self.ref_model = move_model_params_and_grads_to(self.ref_model, "cpu")
        
        # grpo logic
        if self.training_obj == "advantage":
            advantages = torch.clamp(
                samples[0]["advantages"],
                -self.adv_clip_max,
                self.adv_clip_max,
            )
            ratio = torch.exp(log_prob - samples[0]["log_probs"][:, timestep_i])
            # When all successes are False, adv should be set to torch.tensor([-adv_clip_max]), otherwise it should be set to torch.tensor([]).
            advantages = -self.adv_clip_max * torch.ones_like(advantages) if samples[0]["successes"].sum().cpu().item() == 0 else advantages
            unclipped_loss = -advantages.detach() * ratio
            clipped_loss = -advantages.detach() * torch.clamp(
                ratio,
                1.0 - self.clip_range,
                1.0 + self.clip_range,
            )
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        else:
            raise ValueError(f"Unknown training objective: {self.training_obj}")

        if self.kl_weight > 0:
            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
            kl_loss = torch.mean(kl_loss)
            loss = policy_loss + self.kl_weight * kl_loss
            if only_kl_loss:
                loss = kl_loss * self.args.get("only_kl_loss_weight", 0.1)
        else:
            kl_loss = torch.tensor(0.0)
            loss = policy_loss
        
        # TODO: 目前只支持每个rank bs=1的情况：如果batch中没有成功的样本，则直接返回0损失；
        # 不能直接返回tensor(0.0)，因为ds需要保证计算图的完整性
        if samples[0]["successes"].sum().cpu().item() == 0:
            loss = loss * 0.0

        loss_dict = {
            "loss": loss,
            "reward_dummyloss": samples[0]["rewards"],
            "kl_loss": kl_loss,
            "policy_loss": policy_loss,
        }
        # 多个rewards model训练时，用来记录当前rank每个reward model是否用到了，便于打log
        all_rewards_exist_dict = {}
        
        # Add all reward losses with dummy values first
        for k in self.all_reward_loss_keys:
            loss_dict[k] = torch.zeros_like(loss, dtype=torch.float)
            # Convert tensor to list for easier gathering
            all_rewards_exist_dict[k] = False
        
        if self.training_obj == "advantage":
            loss_dict["advantages_dummyloss"] = samples[0]["advantages"]
            loss_dict["unclipped_loss"] = unclipped_loss
            loss_dict["clipped_loss"] = clipped_loss

        # Update reward dict with actual values
        for k, v in samples[0]["rewards_dict"].items():
            dummy_k = f"{k}_dummyloss"
            if dummy_k in self.all_reward_loss_keys:
                loss_dict[dummy_k] = v
                # Convert tensor to list for easier gathering
                all_rewards_exist_dict[dummy_k] = True

        torch.cuda.synchronize()
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        return loss_dict, cur_batch_size, n_tokens, times, samples, all_rewards_exist_dict
    
    def train_loop(self):
        args = self.args
        self.model_engine.train()
        self.ss.current_run_update_steps = 0

        if args.init_save:
            save_checkpoint(args, self.rank, self.logger, self.model_engine, self.ema, self.ss, self.ckpt_dir)

        # Training loop
        start_epoch = self.ss.epoch
        finished = False
        nan_grad_count = 0
        if self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
            # Initialize grpo training states
            self.grpo_states = GRPOTrainingStates(
                iters_per_group=self.train_iters_per_timesteps_group,
                group_size=self.timesteps_group_size,
                max_timesteps=self.num_train_timesteps,
                sample_strategy=self.training_strategy,
                overlap=self.timesteps_group_overlap,
                stride=self.mixgrpo_stride,
            )
            if self.training_strategy == "decay":
                self.grpo_states.set_params(self.decay_kwargs)
            elif self.training_strategy == "dynamic":
                self.grpo_states.set_params(self.dynamic_kwargs)

        for epoch in range(start_epoch, args.max_epochs):
            self.shuffle_dataset_and_set_start_index(self.ss)

            with profiler_context(
                args.profile, self.exp_dir, worker_name=f"Rank_{self.rank}"
            ) as prof:
                self.logger.info(f"Beginning epoch {epoch}...")
                try:
                    self.logger.info(f"  Steps left this epoch: {len(self.dataloader) // self.grad_accu_steps:,}")
                except NotImplementedError:
                    pass
                # Define cycle states, which accumulate the training information between log_steps.
                cs = self.get_states_cls('cycle')()
                torch.cuda.synchronize()
                start_time = time.time()
                data_start = time.time()
                times = {}

                for bi, batch in enumerate(self.dataloader):
                    torch.cuda.synchronize()
                    times['data'] = time.time() - data_start

                    # Dry run dataloader to check data processing.
                    if args.get('dry_run_dataloader'):
                        if bi > 0 and bi % 20 == 0:
                            self.logger.info(
                                f"Dry run dataloader: {bi} batches processed. Average time: {times['data'] / 20:.2f}s."
                            )
                            data_start = time.time()

                        continue
                    samples = None
                    batch_size = None
                    n_tokens = None
                    if self.training_strategy == "all":
                        timesteps_train = [ti for ti in range(self.num_train_timesteps)]
                    elif self.training_strategy in ["progressive", "random", "decay", "dynamic"]:
                        timesteps_train = self.grpo_states.get_current_timesteps()
                        only_kl_loss = False
                        self.grpo_states.update_iteration(seed=batch["seeds"][0] if self.training_strategy == "random" else None)

                    if self.use_extra_low_timesteps_kl:
                        extra_kl_start_timestep = self.args.get("extra_kl_start_timestep", 20)
                        extra_kl_end_timestep = self.args.get("extra_kl_end_timestep", 29)
                        timesteps_train = timesteps_train + [
                            random.randint(
                                extra_kl_start_timestep,
                                extra_kl_end_timestep,
                            )
                        ]

                    for timestep_idx in timesteps_train:
                        if timestep_idx >= extra_kl_start_timestep and timestep_idx <= extra_kl_end_timestep and self.use_extra_low_timesteps_kl:
                            only_kl_loss = True
                        elif self.use_extra_low_timesteps_kl:
                            only_kl_loss = False

                        (
                            loss_dict,
                            batch_size,
                            n_tokens,
                            forward_times,
                            samples,
                            all_rewards_exist_dict
                        ) = self.train_step(
                            batch,
                            timestep_idx,
                            samples,
                            batch_size,
                            n_tokens,
                            timesteps_train,
                            only_kl_loss,
                        )
                        self.logger.info(f"rank-{self.rank}, timestep-{timestep_idx}, loss: {loss_dict['loss']}")
                        times.update(forward_times)

                        backward_start = time.time()
                        loss = loss_dict["loss"].mean()
                        for k, v in loss_dict.items():
                            if "loss" in k and k != "loss":
                                cs.running_sub_loss_dict[k] += v.mean().item()
                                # Only update loss if it exists in current rank
                                # FIXME: 仅支持batchsize=1的情况
                                if k not in all_rewards_exist_dict or (k in all_rewards_exist_dict and all_rewards_exist_dict[k]):
                                    # cs.running_sub_loss_dict[k] += v.mean().item()
                                    cs.running_sub_step_dict[k] += 1
                        self.model_engine.backward(loss)
                        torch.cuda.synchronize()
                        times['backward'] = time.time() - backward_start
                        is_update_step = self.update_train_states(self.ss, cs, batch, batch_size, n_tokens, loss.item())

                        if args.skip_nan_grad and hasattr(self.model_engine.optimizer, "scaled_global_norm"):
                            scaled_grad_norm = self.model_engine.optimizer.scaled_global_norm()     
                            if torch.any(torch.isnan(scaled_grad_norm)):
                                nan_grad_count += 1
                                self.logger.info(f"Step {self.ss.update_steps:07d} grad norm is nan, skipping step. Total nan grad count: {nan_grad_count}.")
                                self.model_engine.optimizer.zero_grad()

                        # Update model parameters at the boundary of gradient accumulation.
                        update_start = time.time()
                        # Get the lr before optimizer.step()
                        lrs = [group["lr"] for group in self.optimizer.param_groups]
                        self.model_engine.step(lr_kwargs={"last_batch_iteration": self.lr_helper(self.ss.update_steps)})
                        torch.cuda.synchronize()
                        times['update'] = time.time() - update_start

                        if self.ss.update_steps >= args.max_training_steps:
                            # Enter stopping routine if max steps reached after this step.
                            finished = True

                        # Update EMA model at the step of main model parameters update.
                        if args.use_ema and is_update_step:
                            self.ema.update(self.model_engine.module)

                        # Log training information:
                        if is_update_step and self.ss.update_steps % args.log_every == 0:
                            # All-gather scalar states and cycle states.
                            all_cs: List[Optional[CycleStates]] = [None for _ in range(self.world_size)]
                            torch.distributed.all_gather_object(all_cs, cs)
                            all_rewards_exist_dicts: List[Optional[Dict[str, bool]]] = [None for _ in range(self.world_size)]
                            torch.distributed.all_gather_object(all_rewards_exist_dicts, all_rewards_exist_dict)

                            # Calculate average main loss
                            avg_loss = sum([cs_i.running_loss for cs_i in all_cs]) / sum([cs_i.log_steps for cs_i in all_cs])
                            
                            # Calculate average sub losses based on exist flags
                            merged_loss_dict = {}
                            merged_step_dict = {}
                            for k in sorted(cs.running_sub_loss_dict.keys()):
                                total_loss = 0
                                total_steps = 0
                                exist_count = 0
                                for rank_idx, (cs_i, exist_dict) in enumerate(zip(all_cs, all_rewards_exist_dicts)):
                                    if k in cs_i.running_sub_loss_dict:
                                        # For reward losses, check exist flag; for other losses, always include
                                        should_include = True
                                        if k in exist_dict:
                                            should_include = exist_dict[k]
                                        
                                        if should_include:
                                            total_loss += cs_i.running_sub_loss_dict[k]
                                            total_steps += cs_i.running_sub_step_dict[k]
                                            exist_count += 1
                                
                                if exist_count > 0:  # Only include if at least one rank has this loss
                                    merged_loss_dict[k] = total_loss
                                    merged_step_dict[k] = total_steps
                            
                            avg_sub_loss_dict = {k: merged_loss_dict[k] / merged_step_dict[k] for k in merged_loss_dict}
                            # Calculate cumulated metrics.
                            cum_samples = self.update_log_states(self.ss, all_cs)

                            # Synchronize cuda to accurately measure training speed:
                            torch.cuda.synchronize()
                            end_time = time.time()
                            steps_per_sec = cs.log_steps / self.grad_accu_steps / (end_time - start_time)
                            seconds_per_step = (end_time - start_time) / (cs.log_steps / self.grad_accu_steps)
                            samples_per_sec = cum_samples / (end_time - start_time)

                            grad_norm = self.model_engine.get_global_grad_norm()
                            user_log_events, user_summary_events = self.get_events(self.ss, avg_loss)

                            log_events = [
                                f"Train Loss: {avg_loss:.4f}",
                                *[f"{k}: {v:.4f}" for k, v in avg_sub_loss_dict.items()],
                            ] + [f"Lr{lr_i}: {lr:.6g}" for lr_i, lr in enumerate(lrs)] + [
                                f"Steps/Sec: {steps_per_sec:.2f}",
                                f"Sec/Step: {seconds_per_step:.2f}",
                                f"Samples/Sec: {int(samples_per_sec):d}",
                                f"Global Grad Norm: {grad_norm:.4f}",
                                f"Nan Grad Count: {nan_grad_count}",
                            ] + user_log_events + [
                                f"T{time_key}: {duration:.4f}"
                                for time_key, duration in times.items()
                            ]
                            summary_events = [
                                ("Train/Steps/train_loss", avg_loss, self.ss.update_steps),
                                *[("Train/Steps/" + k, v, self.ss.update_steps) for k, v in avg_sub_loss_dict.items()],
                                ("Train/Steps/LR", self.ss.lr, self.ss.update_steps),
                                ("Train/Steps/steps_per_sec", steps_per_sec, self.ss.update_steps),
                                ("Train/Steps/samples_per_sec", int(samples_per_sec), self.ss.update_steps),
                                ("Train/Steps/seconds_per_step", seconds_per_step, self.ss.update_steps),
                                ("Train/Steps/grad_norm", grad_norm, self.ss.update_steps),
                                ("Train/ComputationsAttn/train_loss", avg_loss, self.ss.consumed_computations_attn),
                                ("Train/ComputationsTotal/train_loss", avg_loss, self.ss.consumed_computations_total),
                            ] + user_summary_events
                            # Log the training information to the logger.
                            self.logger.info(f"(step={self.ss.update_steps:07d}) " + ", ".join(log_events))
                            # Log the training information to the monitor.
                            if self.model_engine.monitor.enabled and self.rank == 0:
                                self.model_engine.monitor.write_events(summary_events)

                            # Reset monitoring variables:
                            cs.reset()
                            start_time = time.time()

                    # Save checkpoint:
                    if (is_update_step and self.ss.update_steps % args.ckpt_every == 0) or (
                        finished and args.final_save
                    ):
                        self.save_checkpoint()

                    # Perform evaluation
                    if args.validation_every > 0 and (
                        (is_update_step and self.ss.update_steps % args.validation_every == 0)
                        or (
                            is_update_step
                            and self.ss.current_run_update_steps in args.validation_at_steps
                        )
                        or finished
                    ):
                        # Clear the cache to save GPU memory.
                        torch.cuda.empty_cache()
                        # del loss_dict
                        self.model_engine.module.eval()
                        self.val_logger.info(
                            f"Start evaluation after train epoch={self.ss.epoch}, step={self.ss.update_steps} "
                            + (f"(update_step={self.ss.update_steps:07d}) " if self.grad_accu_steps > 1 else "")
                        )
                        with torch.no_grad():
                            self.eval_step(loss_dict)
                        # Wait for rank 0 finished processing and saving
                        dist.barrier()
                        # Return to training mode
                        self.model_engine.module.train()
                        # Clear the cache to save GPU memory.
                        torch.cuda.empty_cache()

                    if prof:
                        prof.step()

                    if finished:
                        self.logger.info(f"Finished and breaking loop at step={self.ss.update_steps}.")
                        break

                    torch.cuda.synchronize()
                    data_start = time.time()

                if finished:
                    self.logger.info(f"Finished and breaking loop at epoch={epoch}.")
                    break

                # Reset epoch states
                new_epoch = self.ss.inc_epoch()
                self.logger.info(f"Increase epoch to {new_epoch}.")

    def build_sampler(self):
        model_dict = dict(
            model_settings=self.model_settings,
        )
        if self.kl_weight > 0:
            model_dict['ref_model'] = self.ref_model

        # model在推理时(prepare_model_inputs)通过unwarp self.model_engine定义即可
        factor_kwargs = {"device": self.device, "dtype": PRECISION_TO_TYPE[self.args.precision]}
        self.model_dict = GeminiBetaSampler.build_extra_model(
            self.args,
            model_dict,
            factor_kwargs,
            logger=self.logger,
        )
        self.sampler = GeminiBetaSampler(
            self.args,
            model_dict=self.model_dict,
            pipeline_name=self.pipeline_name,
            rank=self.dp_rank,
            world_size=self.dp_size,
            device=self.device,
            logger=self.logger,
        )

    def build_data_iterator(self):
        self.dataloader = CombinedBatchIterator(
            ss=self.ss,
            fast_shuffle=self.args.fast_shuffle,
            rank=self.dp_rank,
            world_size=self.dp_size,
            datasets=self.dataset_dict,
            samplers=self.sampler_dict,
            dataloaders=self.dataloader_dict,
            sampling_probs=self.sampling_probs_dict,
            initial_seed=self.args.global_seed,
            sampling_mode=self.args.get('combined_iterator_sampling_mode', 'random'),
            cache_shuffle=self.args.get('cache_shuffle'),
            fixed_key=self.cur_key,
            fixed_key_group=self.cur_key_group,
            force_sync_shuffle=False,
        )
    
    def build_dataloader(self):
        args = self.args
        self.dataloader_preliminary_setup()

        dataloader_kwargs = dict(
            **args.dataloader_params,
            worker_init_fn=set_worker_seed_builder(self.dp_rank),
            shuffle=False,
            drop_last=True,
        )
        sampler_kwargs = dict(
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
        )

        def _filter_dummies(dummy_list):
            # Filter out dummies that are not in the dataset keys
            valid_dummies = []
            for dummy_candidate in dummy_list:
                if any(task in self.all_dataset_keys for task in self.dummy_to_tasks[dummy_candidate]):
                    valid_dummies.append(dummy_candidate)
            return valid_dummies

        # =====================================
        #     Text(+Image) to image data
        # =====================================
        self.task_dummy_dict['t2i'] = _filter_dummies(['mmu'])
        self.task_dummy_dict['face_id_clip'] = []
        task_info_list = [
            dict(dataset_tag="t2i", cur_task="t2i", cls=RLTextImageArrowStream),
        ]
        for item in task_info_list:
            dataset_tag = item['dataset_tag']
            if dataset_tag not in self.all_dataset_keys:
                continue
            task_batch_size = args.get(f'{dataset_tag}_batch_size', self.micro_batch_size)
            if dataset_tag == "t2i":
                multireso = args["t2i_index_kwargs"]["multireso"]
            else:
                multireso = args.get(f'{dataset_tag}_index_kwargs')[f"{dataset_tag}_multireso"]
            self.dataset_dict[dataset_tag] = item['cls'](
                args=args,
                dataset_tag=dataset_tag,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=dict(
                    batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                    world_size=1,  # Dataset don't need to align with world_size. It will be handled by the sampler.
                    **args.get(f'{dataset_tag}_index_kwargs'),
                ),
                logger=self.logger,
                dummy_number=sum([self.dummy_dict[task] for task in self.task_dummy_dict[item['cur_task']]]),
                template=args.sequence_template,
                # if sequence batch is enabled, attention mask sequence length -1 is disabled in __getitem__,
                # and performed in seq_collate_fn instead.
                attn_mask_seq_m1=dataset_tag not in self.seq_batch_size,
            )
            # Build repeated sampler and data loader
            self.sampler_dict[dataset_tag] = RepeatRandomDistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.dataset_num_replicas[dataset_tag],
                rank=self.dataset_rank[dataset_tag],
                batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                mini_repeat_count=self.num_generations,
                repeat_count=self.num_grpo_iterations,
                **sampler_kwargs,
            )
            self.dataloader_dict[dataset_tag] = DataLoader(
                self.dataset_dict[dataset_tag],
                batch_size=task_batch_size,
                sampler=self.sampler_dict[dataset_tag],
                collate_fn=(self.dataset_dict[dataset_tag].collate_fn
                            if hasattr(self.dataset_dict[dataset_tag], "collate_fn")
                            else None),
                **dataloader_kwargs,
            )
