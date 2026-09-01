import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from .priors import semantic_keys



class ScorerBase(ABC):
    """抽象基类：定义评分策略的接口规范[6,8](@ref)"""
    @abstractmethod
    def compute_scores(self, all_item_list: list) -> dict:
        """计算模型得分并返回结果字典
        :param all_item_list: 从JSON加载的数据列表
        :return: {模型名: 平均分} 的字典
        """
        pass

class EqualWeightedScorer(ScorerBase):
    """具体实现类：无优先级均分计算策略[2](@ref)"""
    def compute_scores(self, all_item_list: list) -> dict:
        # Initialization
        info = all_item_list[0]['img_info_list']
        nr_models = len(info)
        model_names = [
            item['model_name']
            if 'model_name' in item
            else os.path.basename(os.path.dirname(item['local_image_path']))
            for item in info
        ]
        image_roots = [os.path.dirname(item['local_image_path']) for item in info]
        all_dir_results = [
            {
                'image_root': root,
                'model_name': name,
                'stats': {
                    'image_level_accuracy': 0.0,
                    **{k: {'accuracy': []} for k in semantic_keys}
                }
            }
            for i, (name, root) in enumerate(zip(model_names, image_roots))
        ]
        nr_imgs = len(all_item_list)

        # Aggregation: 按语义点分 group 合并所有样本的得分
        for i in range(nr_imgs):
            # [
            #   {
            #     "主要主体-名词": [1, 1],
            #     "主要主体-关键属性": [1],
            #     "主要主体-其他属性": [1, 1, 1],
            #     "主要主体-动作": [1],
            #     "次要主体-名词": [],
            #     "次要主体-属性": [],
            #     "次要主体-动作": [],
            #     "场景-名词": [1],
            #     "场景-属性": [1, 1],
            #     "镜头": [],
            #     "风格": [1],
            #     "构图": [],
            #   }
            # ]
            structured_match_res = all_item_list[i]['structured_semantic_points_matching']
            assert len(structured_match_res) == nr_models
            for j in range(nr_models):
                semantic_key_to_01_discrete_scores = structured_match_res[j]
                for k, score_list in semantic_key_to_01_discrete_scores.items():
                    # eliminate nan
                    score_list = [s for s in score_list if not np.isnan(s)]
                    # 第 j 个模型的第 k 个语义点的得分列表
                    all_dir_results[j]['stats'][k]['accuracy'].extend(score_list)

        def sum_and_len(arr):
            return sum(arr), len(arr)
        
        # Reduction
        for j in range(nr_models):
            all_accuracy_list = []
            sum_, len_ = 0, 0
            for k in semantic_keys:
                localSum, localLen = sum_and_len(all_dir_results[j]['stats'][k]['accuracy'])
                sum_ += localSum
                len_ += localLen
                all_dir_results[j]['stats'][k]['accuracy'] = 0 if localLen == 0 else localSum / localLen
            # `image_level_accuracy` is a bad name. Actually it is the semantic-level (or so called global-level) accuracy.
            all_dir_results[j]['stats']['image_level_accuracy'] = sum_ / len_

        return all_dir_results

class ScorerFactory:
    """评分策略工厂：根据名称创建具体评分器实例[2,5](@ref)"""
    _registry = {
        'equal_weighted': EqualWeightedScorer,
        # 可在此处注册其他评分策略
    }
    
    @classmethod
    def create_scorer(cls, scorer_name: str) -> ScorerBase:
        """创建指定类型的评分器
        :param scorer_name: 评分策略名称
        :return: ScorerBase实例
        """
        scorer_class = cls._registry.get(scorer_name)
        if not scorer_class:
            raise ValueError(f"未知的评分策略: {scorer_name}。"
                             f"可用策略: {list(cls._registry.keys())}")
        return scorer_class()
    
    @classmethod
    def register_scorer(cls, name: str, scorer_class):
        """注册新的评分策略（扩展用）[6](@ref)
        :param name: 策略名称
        :param scorer_class: ScorerBase的子类
        """
        if not issubclass(scorer_class, ScorerBase):
            raise TypeError("评分器必须是ScorerBase的子类")
        cls._registry[name] = scorer_class


def main(args):
    # 加载JSON数据
    with open(args.json_path, 'r') as fid:
        all_item_list = json.load(fid)
    
    # 通过工厂创建评分器
    scorer = ScorerFactory.create_scorer(args.scorer)
    
    # 计算并输出结果
    results = scorer.compute_scores(all_item_list)
    for model, score in results.items():
        print(f'{model}: {score:.4f}')


if __name__ == '__main__':
    """
    用法示例：
    python scorer.py --json_path outputs/output.json --scorer equal_weighted
    """
    parser = argparse.ArgumentParser("模型评分系统", add_help=True)
    parser.add_argument('--json_path', type=str, required=True, help='输入JSON文件路径')
    parser.add_argument('--scorer', type=str, default='equal_weighted',
                        help=f'评分策略（默认：equal_weighted）')
    args = parser.parse_args()
    
    main(args)
