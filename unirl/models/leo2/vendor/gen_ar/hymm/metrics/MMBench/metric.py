import os
from typing import Optional

import pandas as pd

try:
    from vlmeval.dataset import ImageMCQDataset
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"{e}. Please install VLMEvalKit first.")

from ..base_metric import BaseMetric
from ...utils.file_utils import safe_file


class MMBenchMetric(BaseMetric):
    def __init__(self, LMUDataRoot, dataset_name="MMBench_DEV_EN"):
        super().__init__()
        self.dataset_name = dataset_name
        os.environ["LMUData"] = LMUDataRoot
        self._dataset: Optional[ImageMCQDataset] = None
        self.results = []
        # {} for timestamp
        self.save_file_template = f"{dataset_name}_{{}}.xlsx"
        # Specify mandatory arguments for compute_metrics
        self.compute_metrics_required_args = ['save_file']

    def load_model(self, logger=None):
        self._dataset = ImageMCQDataset(self.dataset_name)
        if logger is not None:
            logger.info("MMBenchMetric: loaded.")

    def release_model(self):
        self._dataset = None

    def process(self, answers, ids, lines: pd.DataFrame, **kwargs):
        lines = lines.copy(deep=True)
        lines['prediction'] = answers
        self.results.append(lines)

    def compute_metrics(self, results, save_file=None):
        assert save_file is not None, \
            f"MMBenchMetric requires a `save_file` to save the results when computing metrics."
        # Merge results into a single dataframe
        results = pd.concat(results, ignore_index=True)
        results.to_excel(safe_file(save_file), index=False)

        eval_results = self._dataset.evaluate(str(save_file))
        overall = eval_results.loc[0, 'Overall']

        return float(overall), int(len(results))
