from transformers import AutoModelForCausalLM
from typing import List, Dict, Tuple



def flatten_structured_info_into_list(
        structured_info_of_cur_sentence: Dict[str, List[str]]
    ) -> Tuple[List[str], Dict[int, str]]:
    '''
    把根据某种类目分级体系对prompt解析成的结构化信息，给拍平成 list of semantic points 的形式

    Input:
        structured_info_of_cur_sentence: 结构化信息
    '''

    sentence_idx_to_taxonomy_key = dict()
    phrases_of_certain_sentence = []
    idx = 0
    for cls, sentence_list in structured_info_of_cur_sentence.items():
        for sentence in sentence_list:
            phrases_of_certain_sentence.append(sentence)
            sentence_idx_to_taxonomy_key[idx] = cls
            idx += 1
    return phrases_of_certain_sentence, sentence_idx_to_taxonomy_key


def convert_to_structured_matching_from_flattend_matching(
        flattened_semantic_points_matching:  List[List[int]],
        flattened_point_idx_to_taxonomy_key: Dict[int, str],
        structured_semantic_points:          Dict[str, List[str]],
        flattened_semantic_points:           List[str]
    ) -> List[Dict[str, List[int]]]:
    '''
    Inputs:
        flattened_semantic_points_matching, (N, K):
            N张图对平摊形式的K个语义点(拆分自某prompt)的匹配情况, 置为0/1/nan, nan是因为调用某次 "图-语义是否匹配" 的api任务失败了, 客户在外面注意ignore nan的case即可
        flattened_point_idx_to_taxonomy_key 
            记录平摊情况下, 语义点的索引对应的结构化的形式的一级类目名称, 但信息不是完备的，同属于一个key对应的语义点的相互之间的索引顺序可能不同, 因此需要 flatten_idx和point的关系，以及原始的
        structured_semantic_points:
            原始prompt结构化拆分的情况, 注意当前仅支持一级类目
        flattened_semantic_points, (K,):
            结构化拆分的平摊版本
    Return:
        structured_semantic_points_matching:  N张图，每张图对于结构化组织的语义点的匹配情况
    '''
    
    structured_semantic_points_matching = list()
    taxonomy_keys = set(flattened_point_idx_to_taxonomy_key.values())
    
    for img_idx, flattened_matching in enumerate(flattened_semantic_points_matching):
        # flattened_matching: (K,)
        assert len(flattened_matching) == len(flattened_point_idx_to_taxonomy_key.keys())
        assert len(flattened_matching) == len(flattened_semantic_points)
        structured_alignment = {cls: [] for cls in taxonomy_keys}
        point_to_align_res = dict()
        for point_idx, (point, align_res) in enumerate(zip(flattened_semantic_points, flattened_matching)):
            point_to_align_res[point] = align_res
        structured_matching_cur_img = dict()
        for taxonomy, list_of_sub_points in structured_semantic_points.items():
            structured_matching_cur_img[taxonomy] = [point_to_align_res[pt] for pt in list_of_sub_points]
        structured_semantic_points_matching.append(structured_matching_cur_img)
    return structured_semantic_points_matching


def load_model(model_id = "Qwen/Qwen2-7B-Instruct", device = "cuda:0"):
    pipeline = dict()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype="auto",
        device_map=device
        #device_map="auto"
    )
    pipeline['model'] = model
    return pipeline
