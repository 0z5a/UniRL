from pathlib import Path

import torch
from loguru import logger

from hymm.trainers.helpers import MultiModalScalarStates
from hymm.trainers.multimodal_gemini_beta_trainer import MultiModalGeminiBetaTrainer
from megatron import mpu

# _GLOBAL_GEMINI_TRAINER initialize dataloader, extra_model and data_iterator
_GLOBAL_GEMINI_TRAINER = None


def get_gemini_trainer():
    assert _GLOBAL_GEMINI_TRAINER is not None, '_GLOBAL_GEMINI_TRAINER is not initialized.'
    return _GLOBAL_GEMINI_TRAINER


class GeminiBetaTrainerPTMWrapper(MultiModalGeminiBetaTrainer):
    # noinspection PyMissingConstructor
    def __init__(self, args):
        """
        prepare_model_inputs need self.dummy_dict, self.dataset_dict, self.task_dummy_dict depends on build_dataloader
        """
        self.raw_args = args
        # Convert Namespace to EasyDict for uniform access
        # self.args = EasyDict(vars(args))
        self.args = args
        # adapt megatron arguments args.seed
        self.args.global_seed = self.args.seed

        # mock init members
        self.dataset = None
        self.world_size = mpu.get_data_parallel_world_size()
        self.rank = mpu.get_data_parallel_rank()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_size = mpu.get_data_parallel_world_size()
        self.device = torch.device("cuda", args.local_rank)
        self.logger = logger
        self.ss = MultiModalScalarStates()
        self.micro_batch_size = args.micro_batch_size
        # self.global_batch_size = args.global_batch_size
        self.exp_dir = Path(args.save).parent

        # self.sampling_probs_dict, self.all_dataset_keys, self.dummy_to_tasks, self.seq_batch_size, 
        self.task_init(self.args)

    def get_pp_rank(self):
        return mpu.get_pipeline_model_parallel_rank()

    def get_pp_world_size(self):
        return mpu.get_pipeline_model_parallel_world_size()

    def get_data_iterator(self):
        # same with build_data_iterator in MultiModalGeminiBetaTrainer
        self.build_data_iterator(
            # use_ptm enable ptm_dp_size and ptm_dp_group, so shuffle_flags all_gather in ptm_dp_group
            use_ptm=True,
            ptm_dp_size=self.dp_size,
            ptm_dp_group=mpu.get_data_parallel_group(),
        )
        return self.dataloader


def initialize_gemini_trainer(dict_args):
    global _GLOBAL_GEMINI_TRAINER
    assert _GLOBAL_GEMINI_TRAINER is None, '_GLOBAL_GEMINI_TRAINER is already initialized.'
    _GLOBAL_GEMINI_TRAINER = GeminiBetaTrainerPTMWrapper(dict_args)
    return _GLOBAL_GEMINI_TRAINER
