import json
import tempfile

from .vqa import VQA
from .vqa_eval import VQAEval
from ..base_metric import BaseMetric
from ...constants import VQA_DATA_PATH


class VQAScoreMetric(BaseMetric):
    def __init__(self, dataset_name="vqav2_val"):
        super().__init__()
        self.dataset_name = dataset_name
        self.question_path = VQA_DATA_PATH[dataset_name]['question']
        self.annotation_path = VQA_DATA_PATH[dataset_name]['annotation']
        self.results = []

    def load_model(self, logger=None):
        if logger is not None:
            logger.info("VQAScoreMetric: loaded.")

    def release_model(self):
        pass

    def process(self, answers, ids, **kwargs):
        for answer, id_ in zip(answers, ids):
            self.results.append({
                'question_id': int(id_),
                'answer': str(answer),
            })

    def compute_metrics(self, results, **kwargs):
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            results_file = tmp_file.name
            json.dump(results, tmp_file, ensure_ascii=False)
            print("VQAScoreMetric: results file is", results_file)

        vqa = VQA(self.annotation_path, self.question_path)
        vqa_results = vqa.loadRes(resFile=results_file, quesFile=self.question_path)
        vqa_scorer = VQAEval(vqa, vqa_results, n=2)
        vqa_scorer.evaluate()

        return float(vqa_scorer.accuracy['overall']), int(len(results)), dict(
            extra_info_dict=vqa_scorer.accuracy,
        )
