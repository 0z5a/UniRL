# ================================================================
# This module is part of the package `hymm`.
# We collect all the metric classes in this file for easy access.
# ================================================================


def load_metric(metric_name, default_max_size=65536, logger=None, **kwargs):
    """
    metric_name pattern: {metric_type}[@{dataset_name}[@{max_size}]]

    Example metric names:
      - fid@coco6k@256
      - fid@coco30k@256
      - clip_score@coco6k@256
      - clip_score@coco30k@256
      - t2i_compbench
      - dpg_bench
      - geneval
      - cider
      - counting_eval
      - mmlu_bench
      - mmlu_pro_bench
    """
    metric_type, *res = metric_name.split('@')
    dataset_name = res[0] if len(res) > 0 else metric_type
    max_size = int(res[1]) if len(res) > 1 else default_max_size
    if len(res) > 2:
        if logger is None:
            from loguru import logger
        logger.warning(f"Unknown metric arguments: {res[2:]}")

    if metric_type == 'fid':
        from .fid.metric import FIDMetric
        from ..constants import FID_INCEPTION_PATH, FID_TARGET_PATH

        metric = FIDMetric(dims=2048,
                           inception_path=FID_INCEPTION_PATH,
                           target_path=FID_TARGET_PATH[dataset_name],
                           dataset_name=dataset_name,
                           max_size=max_size,
                           )

    elif metric_type == 'clip_score':
        from .clip_score.metric import CLIPScoreMetric
        from ..constants import CLIP_MODEL_PATH

        metric = CLIPScoreMetric(clip_model_path=CLIP_MODEL_PATH,
                                 dataset_name=dataset_name,
                                 max_size=max_size,
                                 )

    elif metric_type == 'hpsv2':
        from .hpsv2.metric import HPSv2Metric
        from ..constants import HPSV2_MODEL_PATH

        metric = HPSv2Metric(hpsv2_model_path=HPSV2_MODEL_PATH,
                             dataset_name=dataset_name,
                             max_size=max_size,
                             )

    elif metric_type == 't2i_compbench':
        from .t2i_compbench.metric import T2ICompBenchMetric
        from ..constants import VQA_MODEL_PATH

        metric = T2ICompBenchMetric(vqa_model_path=VQA_MODEL_PATH,
                                    dataset_name=dataset_name,
                                    max_size=max_size,
                                    )

    elif metric_type == 'dpg_bench':
        # DPGBenchMetric has extra dependencies, so we import it here.
        from .dpg_bench.metric import DPGBenchMetric
        from ..constants import MPLUG_MODEL_PATH

        metric = DPGBenchMetric(vqa_model_path=MPLUG_MODEL_PATH,
                                dataset_name=dataset_name,
                                max_size=max_size,
                                )

    elif metric_type == 'geneval':
        from .geneval.metric import GenEvalMetric
        from ..constants import GENEVAL_MODEL_PATH

        metric = GenEvalMetric(model_path=GENEVAL_MODEL_PATH,
                               dataset_name=dataset_name,
                               max_size=max_size,
                               )
    
    elif metric_type == 'counting_eval':
        from .counting_eval.metric import CountingEvalMetric
        
        metric = CountingEvalMetric(dataset_name=dataset_name,
                                    max_size=max_size,
                                    )

    elif metric_type.startswith("cider"):
        from .CIDEr.metric import CIDErMetric
        from ..constants import CIDER_TOKENIZER

        metric = CIDErMetric(model_path=CIDER_TOKENIZER,
                             dataset_name=dataset_name,
                             metric_type=metric_type)

    elif metric_type == 'vqa_score':
        from .VQA.vqa_score_metric import VQAScoreMetric

        metric = VQAScoreMetric(dataset_name=dataset_name)

    elif metric_type == 'mmbench':
        from .MMBench.metric import MMBenchMetric
        from ..constants import LMUDataRoot

        metric = MMBenchMetric(LMUDataRoot=LMUDataRoot,
                               dataset_name=dataset_name)

    elif metric_type == 'mmmu':
        from .MMMU.metric import MMMUMetric

        metric = MMMUMetric(dataset_name=dataset_name)

    elif metric_type == 'mmlu_bench':
        from .mmlu.metric import MMLUMetric
        
        metric = MMLUMetric(dataset_name=dataset_name,
                            prefix_space=kwargs.get('prefix_space', False),
                            **kwargs)
    
    elif metric_type == 'mmlu_pro_bench':
        from .mmlu_pro.metric import MMLUProMetric

        metric = MMLUProMetric(dataset_name=dataset_name,
                               **kwargs)

    elif metric_type == "imagebind_score":
        from .imagebind.metric import ImageBindMetric
        from ..constants import IMAGEBIND_MODEL_PATH
        metric = ImageBindMetric(image_bind_path=IMAGEBIND_MODEL_PATH, dataset_name=dataset_name)
    elif metric_type == "clap_score":
        from .clap_score.metric import ClapScoreMetric
        from ..constants import CLAP_MODEL_PATH
        metric = ClapScoreMetric(clap_model_path=CLAP_MODEL_PATH, dataset_name=dataset_name)
    elif (metric_type == "fad" or metric_type == "fad_vggish"):
        from .fad.metric import FADMetric
        from ..constants import FAD_TARGET_PATH, FAD_VGGISH_TARGET_PATH, CLAP_MODEL_PATH, VGGISH_MODEL_PATH, VGGISH_PCA_PATH
        if metric_type == "fad_vggish":
            model_type = "vggish"
            model_path = VGGISH_MODEL_PATH
            pca_path = VGGISH_PCA_PATH
            fad_gt = FAD_VGGISH_TARGET_PATH[dataset_name]
        else:
            model_type = "clap"
            model_path = CLAP_MODEL_PATH
            pca_path = None
            fad_gt = FAD_TARGET_PATH[dataset_name]
        metric = FADMetric(
            model_path=model_path, 
            fid_target_path=fad_gt,
            vggish_pca_path=pca_path,
            dataset_name=dataset_name,
            model_type=model_type,
        )
    elif metric_type == "face_sim":
        from .face_sim.metric import FaceSimMetric
        from ..constants import FACE_MODEL_PATH
        metric = FaceSimMetric(
            face_model_path=FACE_MODEL_PATH,
            dataset_name=dataset_name,
        )
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
    return metric
