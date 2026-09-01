import os
import json
import argparse
from pathlib import Path
from jinja2 import Environment, BaseLoader


def get_data(path, tag):
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif os.path.isdir(path):
        root = path
        json_paths = [os.path.join(root, name) for name in os.listdir(root) if name.endswith('.json')]
        data = list()
        for pth in json_paths:
            with open(pth, 'r', encoding='utf-8') as fid:
                data.extend(json.load(fid))
    else:
        raise ValueError(f"Unrecognized input: {path}")

    for item in data:
        for i in range(len(item['img_info_list'])):
            item["img_info_list"][i]["tag"] = tag if tag is not None else item["img_info_list"][i]["model_name"]
    return data


def merge_data(base, data):
    prompt_idx_to_item_idx = {}
    for i, item in enumerate(base):
        prompt_idx_to_item_idx[item['prompt_idx']] = i

    for item in data:
        prompt_idx = item['prompt_idx']
        if prompt_idx in prompt_idx_to_item_idx:
            base_idx = prompt_idx_to_item_idx[prompt_idx]
            base_item = base[base_idx]
            base_item['img_info_list'].extend(item['img_info_list'])
            base_item["semantic_points_matching_request_id"].extend(item['semantic_points_matching_request_id'])
            base_item["semantic_points_matching_details"].extend(item['semantic_points_matching_details'])
            base_item['masked_equally_weighted_score'].extend(item['masked_equally_weighted_score'])
            base_item["structured_semantic_points_matching"].extend(item['structured_semantic_points_matching'])
    return base


def get_all_data(paths, tags):
    base = get_data(paths[0], tags[0])
    if len(paths) > 1:
        for pth, tag in zip(paths[1:], tags[1:]):
            data = get_data(pth, tag)
            base = merge_data(base, data)
    return base


def generate_html(json_paths, tags, output_path="output.html"):
    """生成静态HTML文件

    Args:
        json_path: JSON文件路径或包含JSON文件的目录
        output_path: 输出的HTML文件路径
    """
    # 加载数据
    data = get_all_data(json_paths, tags)

    # 定义HTML模板字符串（直接从原index.html复制）
    template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">  <!-- 关键修复：添加字符集声明 -->
    <title>图像-描述匹配可视化</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 2400px; 
            margin: 20px auto; 
            line-height: 1.6;
        }

        .group-container { 
            margin-bottom: 40px; 
            border-bottom: 2px solid #ddd; 
            padding-bottom: 20px; 
        }

        /* 水平滚动容器 */
        .image-row-container {
            overflow-x: auto; /* 显示水平滚动条 */
            white-space: nowrap; /* 防止子元素换行 */
            margin: 20px 0;
        }

        /* 图片行容器 */
        .image-row {
            display: inline-block; 
            width: fit-content; 
            padding: 0;
        }

        /* 单个图片单元 */
        .image-unit {
            display: inline-block; 
            width: 500px; /* 固定每个单元的宽度 */
            vertical-align: top; 
            margin-right: 20px; 
            border: 1px solid #eee;
            padding: 15px;
            text-align: center;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* 图片样式 */
        .image-unit img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 4px;
        }

        /* 索引样式 */
        .index { 
            margin: 10px 0; 
            font-weight: bold; 
            color: #333;
        }

        /* 描述项容器 */
        .descriptions {
            margin: 15px 0;
            padding: 10px;
            background-color: #f8f8f8;
            border-radius: 4px;
        }

        /* 单个描述项 */
        .description-item {
            margin: 5px 0;
            padding: 5px;
            border-radius: 4px;
        }

        /* 采用/未采用样式 */
        .used { 
            background-color: #f0f8ff; 
            text-decoration: none; 
            color: black;
        }
        .unused { 
            text-decoration: line-through; 
            color: #888; 
            background-color: #f8f8f8;
        }

        /* 分数样式 */
        .score { 
            font-size: 1.2em; 
            color: #e84118; 
            margin: 15px 0; 
            font-weight: bold;
        }

        /* 底部信息 */
        .footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
        }

        .score-certainty {
            font-weight: bold;
            margin-right: 20px;
        }

        .recommended-list {
            list-style: none;
            padding-left: 0;
        }
    </style>
</head>
<body>
    {% for info in data %}
    <div class="group-container">
        <div class="group-header">
            <h2>prompt index: {{ info.prompt_idx}}</h2>
            <p>Prompt: {{ info.prompt }}</p>
        </div>

        <!-- 图片行容器 -->
        <div class="image-row-container">
            <div class="image-row">
                {% for i in range(info.img_info_list|length) %}
                <div class="image-unit">
                    <img src="{{ info.img_info_list[i].url_cos }}">
                    <div class="index">{{ info.img_info_list[i].tag }}</div>

                    <!-- 结构化语义点展示 -->
                    <div class="descriptions">
                        {% for category, points in info.structured_semantic_points.items() if points %}
                            {% for j in range(points|length) %}
                            <div class="description-item {{ 'used' if info.exact_scoring_points_mask[points[j]] else 'unused' }}">
                                {{ category }}: {{ points[j] }}
                                {% if info.exact_scoring_points_mask[points[j]] %}
                                    {% if info.structured_semantic_points_matching[i][category][j] == 1 %}
                                        <span style="color:green; margin-left:5px;">✅</span>
                                    {% else %}
                                        <span style="color:red; margin-left:5px;">❌</span>
                                    {% endif %}
                                {% endif %}
                            </div>
                            {% endfor %}
                        {% endfor %}
                    </div>

                    <div class="score">Score: {{ info.masked_equally_weighted_score[i] | round(2) }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- 底部信息 -->
        <div class="footer">
            <div class="score-certainty">
                Score Certainty: {{ info.score_certainty | round(2) }}
            </div>
        </div>
    </div>
    {% endfor %}
</body>
</html>
    """

    # 创建Jinja2环境并渲染模板
    env = Environment(loader=BaseLoader())
    env.filters['round'] = round  # 注册round过滤器
    template = env.from_string(template_str)
    html_content = template.render(data=data)

    # 写入HTML文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(html_content)

    print(f"静态HTML文件已生成: {output_path}")


if __name__ == '__main__':
    '''
    Usage: python generate_html.py --json_path outputs/output.json --output results.html
    '''
    parser = argparse.ArgumentParser("生成静态HTML可视化", add_help=True)
    parser.add_argument('--json_path', type=str, required=True,
                        help='JSON文件路径或包含JSON文件的目录')
    parser.add_argument('--output', type=str, default='visualization.html',
                        help='输出的HTML文件路径(默认: visualization.html)')
    args = parser.parse_args()

    generate_html(args.json_path, args.output)
