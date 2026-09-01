## 测试环境

* 镜像：mirrors.tencent.com/multimodal_gen_ar/tlinux3.2-cuda12.5-cudnn9-python3.12-torch2.5.1-tccl:v1_libs0312
* 额外依赖： pp 依赖 torch > 2.6.0


## 依赖安装

### 手动安装
手动安装适用于大部分库都装好了，手动装个别的场景

* pytorch 2.6 安装

```bash
echo "export LD_LIBRARY_PATH=\"/usr/local/lib/python3.12/site-packages/cusparselt/lib/\":\$LD_LIBRARY_PATH" >> ~/.zshrc
echo "export LD_LIBRARY_PATH=\"/usr/local/lib/python3.12/site-packages/cusparselt/lib/\":\$LD_LIBRARY_PATH" >> ~/.bashrc
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/cusparselt/lib/":$LD_LIBRARY_PATH
source  ~/.zshrc

yes | pip3 uninstall torchvision torchaudio
pip3 install torch==2.6.0 torchvision torchaudio
yes | pip3 uninstall flash_attn
pip3 install flash_attn==2.7.3 --no-build-isolation --no-cache-dir
```

* 其他依赖安装
```bash
############   PureTorch 相关 ##############
pip3 install cos-python-sdk-v5
pip3 install transformers==4.53.0
pip3 install peft
pip3 install einx
pip3 install colt5_attention
###########################################
```

### 自动安装完整依赖

安装完整依赖 `submit_jobs/install_libs.sh`


### 参数转换

预训练模型转换成 dcp 模型

```python
def gemini80b_transform(hf_model_path, dcp_save_path):
    from hymm.parallelism.checkpoint_manager import torch_state_dict_to_dcp, dcp_to_torch_state_dict
    from hymm.parallelism.moe.ckpt_utils import read_hf_file, hf_to_torch, GeminiDefaultToGeminiEP
    weights = read_hf_file(hf_model_path)
    weights_torch = hf_to_torch(weights)
    weights_torch_ep = GeminiDefaultToGeminiEP().transform(weights_torch)
    torch_state_dict_to_dcp(dcp_save_path, sd_input=weights_torch_ep)
```

### MOE 推理

如果需要单机推理，要先设置下面环境变量，否则无需设置，默认使用 `/etc/taiji/hostfile`
```bash
echo "$LOCAL_IP" slots=8 > /root/localhostfile
export hostfile=/root/localhostfile
```

根据自己的环境设置下面的环境变量：
```bash
#export DS_ENV_FILE==${PROJECT_BASE}/.deepspeed_env

export TOKENIZERS_PARALLELISM=false
export ASSETS_BASE=/apdcephfs_zwfy/share_303937731/1_public_models/hymm_ar_assets
export LOGURU_COLORIZE=true

ckpt_path=/apdcephfs_zwfy/share_303937731/kevinkhwu/pretrain/ep_dcp # 512 训练的ckpt
ckpt_path=/apdcephfs_zwfy/share_303937731/milesjyang/ckpts/stage4_1_step400_torch_dcp # 1024 训练的
save_path=/apdcephfs_cq8/share_2938211/kevinkhwu/tmp/offline_infer_1024
pp_size=8
ep_size=1
image_resolution=1024 # 这个环境变量可能不生效，尺寸以 yaml 为准
# 注意，出于兼容性考虑，MOE 模型 部份 key 没有加入到yaml中，如果不加入会加载ckpt失败，需要手动加入下面参数到 model_config
#  unet_out_norm: True
#  image_preln: True
#  use_final_time_embed: True
config_file=hymm/configs/gemini/hunyuan7b_moe_gemini_beta_ptm_mm_pretrain_stage4.yaml
```

推理脚本：

```bash
bash ./jobs/run_sample.sh gb --deepspeed --no-load-pretrained --no-compile --task t2i \
--puretorch-ckpt $ckpt_path \
--no-load-model \
--config-path $config_file \
--sample-batch-size 1 --sample-image-size $image_resolution --guidance-scale 6.0 --diff-infer-steps 20 --skip-exist \
--csv /apdcephfs_nj10/share_301739632/yutaocui/workspace/hunyuan_multimoda_gen_ar/data/test/drawbench.csv \
--sample-save-path $save_path \
--pp-size $pp_size --ep-size $ep_size --launcher pure_torch
```



## GRPO 训练
注意，使用的是 `multimodal_gemini_beta_grpo_puretorchparallel_trainer.MultiModalGeminiBetaGRPOPureTorchParallelTrainer`

同理，注意里面的 hostfile
```bash
export hostfile=/root/4node
pp_size=4
ep_size=8
sh jobs/run_train.sh hymm/configs/gemini/hunyuan7b_gemini_beta_puretorch_grpo.yaml --pp-size $pp_size --ep-size $ep_size --launcher pure_torch --pp-splits "7,9,9,7"
sh jobs/run_train.sh hymm/configs/gemini/yutao.yaml --pp-size $pp_size --ep-size $ep_size --launcher pure_torch --pp-splits "7,9,9,7"
```

## 常见报错
```
RuntimeError: Missing key in checkpoint state _dict: model. language_model. final_layer.model. 1!!
```
* 解决办法：参考文档上半部分中关于yaml配置部份

---


```
missing argument `input_arg` for PipelineStage
```
* 解决办法：参考文档上半部分，更新pytorch

---

```
hspv2 not found
```
* `submit_jobs/install_libs.sh` 里面有这个的安装脚本，之所以安装失败可能是没有挂载nj8

---
