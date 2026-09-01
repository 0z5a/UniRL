# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Support different forms of monitoring such as wandb and tensorboard
"""
import os
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch.distributed as dist


class Monitor(ABC):

    @abstractmethod
    def __init__(self, monitor_config):
        self.monitor_config = monitor_config

    @abstractmethod
    def write_events(self, event_list):
        pass


@dataclass
class TensorBoardConfig:
    enabled: bool
    output_path: str
    job_name: str


class TensorBoardMonitor(Monitor):

    def __init__(self, tensorboard_config: TensorBoardConfig, writer_rank=0):
        super().__init__(tensorboard_config)
        self.summary_writer = None
        self.enabled = tensorboard_config.enabled
        self.output_path = tensorboard_config.output_path
        self.job_name = tensorboard_config.job_name
        self.writer_rank = writer_rank
        self.tb_executor = None

        if self.enabled and dist.get_rank() == writer_rank:
            self.get_summary_writer()
            self.tb_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tb-writer")
            print(f"[rank {dist.get_rank()}] TensorBoard summary writer initialized at {self.summary_writer.log_dir}")

    def get_summary_writer(self, base=os.path.join(os.path.expanduser("~"), "tensorboard")):
        if self.enabled and dist.get_rank() == self.writer_rank:
            from torch.utils.tensorboard import SummaryWriter
            if self.output_path is not None:
                log_dir = os.path.join(self.output_path, self.job_name)
            # NOTE: This code path currently is never used since the default output_path is an empty string and not None. Saving it in case we want this functionality in the future.
            else:
                if "DLWS_JOB_ID" in os.environ:
                    infra_job_id = os.environ["DLWS_JOB_ID"]
                elif "DLTS_JOB_ID" in os.environ:
                    infra_job_id = os.environ["DLTS_JOB_ID"]
                else:
                    infra_job_id = "unknown-job-id"

                summary_writer_dir_name = os.path.join(infra_job_id, "logs")
                log_dir = os.path.join(base, summary_writer_dir_name, self.output_path)
            os.makedirs(log_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        return self.summary_writer

    def write_events(self, event_list, flush=True):
        event_list = copy.deepcopy(event_list)
        if self.enabled and self.summary_writer is not None and dist.get_rank() == self.writer_rank:
            for event in event_list:
                self.tb_executor.submit(self.summary_writer.add_scalar, *event)
            if flush:
                self.tb_executor.submit(self.summary_writer.flush)

    def flush(self):
        if self.enabled and self.summary_writer is not None and dist.get_rank() == self.writer_rank:
            self.tb_executor.submit(lambda: None).result()
            self.summary_writer.flush()

@dataclass
class WandbConfig:
    enabled: bool
    project: str
    exp_name: str
    output_path: str
    # Track hyperparameters and run metadata.
    config: dict

class WandbMonitor(Monitor):

    def __init__(self, wandb_config: WandbConfig, writer_rank=0, wandb_server=None):
        super().__init__(wandb_config)
        self.wandb_run = None
        self.enabled = wandb_config.enabled
        self.wandb_server = wandb_server
        self.output_path = wandb_config.output_path
        self.project = wandb_config.project
        self.exp_name = wandb_config.exp_name
        self._config = wandb_config.config
        self.writer_rank = writer_rank

        if self.enabled and dist.get_rank() == writer_rank:
            self.init_wandb()
            if self.wandb_run is not None:
                print(f"[rank {dist.get_rank()}] WandB initialized: "
                      f"project={self.project}, exp_name={self.exp_name}, id={self.wandb_run.id}")

    def init_wandb(self):
        """Initialize WandB run"""
        if self.enabled and dist.get_rank() == self.writer_rank:
            # Prepare init arguments
            if self.wandb_server == "swanlab":
                import swanlab as wandb
                wandb_kwargs = {
                    "project": self.project,
                    "experiment_name": self.exp_name,
                    "config": self._config
                }
            else:
                import wandb
                wandb_kwargs = {
                    'dir': self.output_path,
                    "project": self.project,
                    "name": self.exp_name,
                    "config": self._config
                }
                os.makedirs(wandb_kwargs['dir'], exist_ok=True)

            # Initialize WandB
            self.wandb_run = wandb.init(**wandb_kwargs)
        
        return self.wandb_run

    def write_events(self, event_list, flush=True):
        """Write events to WandB"""
        if self.enabled and self.wandb_run is not None and dist.get_rank() == self.writer_rank:      
            for event in event_list:
                tag, value, step = event
                self.wandb_run.log({tag: value}, step)
            
            if flush:
                self.flush()

    def flush(self):
        """Flush WandB logs"""
        if self.enabled and self.wandb_run is not None and dist.get_rank() == self.writer_rank:
            try:
                self.wandb_run.log({})  # Force flush
            except Exception as e:
                print(f"[rank {dist.get_rank()}] Warning: Failed to flush WandB: {e}")

    def finish(self):
        """Finish WandB run"""
        if self.enabled and self.wandb_run is not None and dist.get_rank() == self.writer_rank:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"[rank {dist.get_rank()}] Warning: Failed to finish WandB run: {e}")


class MonitorMaster(Monitor):

    def __init__(self, monitor_config, writer_rank=0, wandb_server=None):
        super().__init__(monitor_config)
        self.tb_monitor = None
        self.wandb_monitor = None
        self.csv_monitor = None
        self.comet_monitor = None
        self.enabled = monitor_config.enabled
        self.writer_rank = writer_rank

        if dist.get_rank() == self.writer_rank:
            if monitor_config.tensorboard.enabled:
                self.tb_monitor = TensorBoardMonitor(monitor_config.tensorboard, writer_rank=writer_rank)
            if monitor_config.wandb.enabled:
                self.wandb_monitor = WandbMonitor(monitor_config.wandb, writer_rank=writer_rank,
                                                  wandb_server=wandb_server)

    def write_events(self, event_list):
        if dist.get_rank() == self.writer_rank:
            if self.tb_monitor is not None:
                self.tb_monitor.write_events(event_list)
            if self.wandb_monitor is not None:
                self.wandb_monitor.write_events(event_list)
            if self.csv_monitor is not None:
                self.csv_monitor.write_events(event_list)
            if self.comet_monitor is not None:
                self.comet_monitor.write_events(event_list)
