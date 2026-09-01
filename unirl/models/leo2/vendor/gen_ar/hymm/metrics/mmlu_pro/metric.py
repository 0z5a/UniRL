import re

import torch
import pandas as pd

from ..base_metric import BaseMetric


INVALID_ANS = "[invalid]"

class MMLUProMetric(BaseMetric):
    def __init__(self, dataset_name="mmlu_pro_bench", report_by_subject=False, device=None, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.results = []
        self.report_by_subject = report_by_subject

        if device is None:
            if torch.distributed.is_initialized():
                device = torch.distributed.get_rank() % 8
            else:
                device = 0
            self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def load_model(self, logger):
        pass

    def _extract_answer(self, text):
        """Extract the answer from the generated text."""
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return self._extract_again(text)

    def _extract_again(self, text):
        """Fallback extraction method."""
        match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
        if match:
            return match.group(1)
        else:
            return self._extract_final(text)

    @staticmethod
    def _extract_final(text):
        """Final fallback extraction method."""
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return INVALID_ANS

    def process(self, answers, labels, subjects, **kwargs):
        response_batch = []
        pred_batch = []

        for output in answers:
            response_batch.append(output)
            pred = self._extract_answer(output)
            pred_batch.append(pred)

        for pred, response, label, subject in zip(pred_batch, response_batch, labels, subjects):
            self.results.append({
                "subject": subject,
                "label": label,
                "pred": pred,
                "response": response,
            })

        return self.results

    def compute_metrics(self, results):
        df = pd.DataFrame(results)
        out_dict = {}

        # Group by subject and calculate accuracy for each subject
        if self.report_by_subject:
            for subject, group in sorted(df.groupby('subject')):
                subject_score = group['pred'].eq(group['label']).mean()
                out_dict[subject] = (round(float(subject_score), 6), len(group))

        # Calculate overall accuracy
        count = len(df)
        avg_accuracy = df['pred'].eq(df['label']).mean()
        out_dict['avg'] = (
            round(float(avg_accuracy), 6),
            count
        )

        return out_dict
