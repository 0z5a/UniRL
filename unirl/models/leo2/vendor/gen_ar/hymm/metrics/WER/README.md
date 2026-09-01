# WER (Word Error Rate)

## 安装开发环境

WER依赖环境如下 (注意和主实验不同)
```shell
# 挂代理
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
# 兼容基于从实验环境生成一个wer-base环境
conda create --name wer-base --clone torch-base
conda activate wer-base

pip install transformers==4.57.6
pip install jiwer
pip install nagisa
```

主实验的transformers版本
```shell
pip install transformers==5.5.4
```

运行评估脚本 (使用参考脚本案例)
```shell
bash eval_wer.sh
```

运行绘图脚本 (使用参考脚本案例)
```shell
python3 plot.py
```