# Hunyuan Auto-Regressive Multimodal Generation

多模态AR范式训练、推理开发代码仓。

## 安装开发环境

参考 [INSTALL.md](INSTALL.md) 文件.

## Git 与 Submodule 使用注意事项

本仓库在 `deps/` 下使用 **Git Submodule** 管理依赖（如 AngelPTM、hy_parallelism、IndexKits、VLMEvalKit 等）。很多人只习惯 `git pull` 或者 `git rebase`，但 **仅执行 `git pull` 不会自动把子模块切换到主仓所记录的 commit**，子模块目录可能仍停留在旧版本，从而出现 import 失败、缺文件、文件找不到路径等报错。**若故障现象与 `deps/` 下代码或第三方路径有关，请先检查子模块是否未更新或未初始化。**

### 首次克隆

优先带 submodule 一起克隆，避免子模块目录为空：

```shell
git clone --recursive <仓库 URL>
```

若已经用普通 `git clone` 拉过代码，在项目根目录执行：

```shell
git submodule update --init --recursive
```

### 日常更新主仓代码（推荐流程）

拉取主仓后，**务必同步子模块**到主仓记录的版本：

```shell
cd <项目根目录>
git pull
git submodule update --init --recursive
```

也可以在一次 `pull` 时顺带更新子模块（需 Git 2.13+，行为与版本有关，以本地 Git 文档为准）：

```shell
git pull --recurse-submodules
```

若希望默认 `git pull` 都尝试处理子模块，可配置:

```shell
git config pull.recurseSubmodules on
# 或
git config submodule.recurse true
```

### 排查子模块是否落后于主仓

如遇到 `hy_parallelism`, `AngelPTM`, `IndexKits`, 等子模块的报错，首先检查所使用分支是否正确，然后检查子模块是否落后于主仓，可以通过执行以下命令查看子模块状态：
在项目根目录执行：

```shell
git submodule status
```

若某子模块前为 `-`，表示尚未初始化；若为 `+`，表示子模块当前 checkout 的 commit **与主仓记录不一致**，通常需要执行上面的 `git submodule update --init --recursive`。
例如：
```
+4bd75fe908553a1e5cbcebd2251a6a594fd09454 deps/AngelPTM (remotes/origin/effort/dev_multimodal_gen-157-g4bd75fe90)
 7240e4c138346e6f84f93b0fd2815566d41e563d deps/IndexKits (v0.3.0-91-g7240e4c)
 34e44549ecb2b264d62600a90f2a472c33d5947d deps/VLMEvalKit (heads/main)
 c405e99c1f22e4fe3fcfe4a9b1ddd85acff8bd19 deps/hy_parallelism (v1.1.7~1)
```


更细的 submodule 说明（含维护者更新子模块的流程）见 [deps/README.md](deps/README.md)。

## 提交太极任务

确认使用的卡型. 目前支持 H800 和 H20.

* H800/H20: `submit_jobs/task.json`

拷贝一份对应卡型的配置到 `submit_jobs/my_task.json` 文件, 修改其中的参数.

|      配置项      |       参数        |                   示例                   |
|:-------------:|:---------------:|:--------------------------------------:|
|     Token     |    填写你的Token    |                 `xxxx`                 |
| business_flag |    GPU 集群标识     |   `TaiJi_HYAide_Text2Video_NJ_H800H`   |
|   host_num    |     申请的节点数量     |                   1                    |
|   task_flag   |     自定义任务名称     |              `hymm_debug`              |
|   start_cmd   | 启动命令, 填写你的Token |  `TOKEN=xxxx sh submit_jobs/start.sh`  |
|   task_flag   |     自定义任务名称     |              `hymm_debug`              |
|   HUNYUAN_TASK_DESCRIPTION   | 任务描述 | `autoregressive text-image experiment` |
|   HUNYUAN_RESOURCE_USAGE   |     资源用途     |              `experiment`              |
|   HUNYUAN_BASE_MODEL   | 使用的基座模型 |                 `base`                 |
|   HUNYUAN_OUTPUT_BASE_MODEL  | 产出的基座模型 |                  `6B`                  |

**在 DevCloud 上**, 执行以下命令提交任务.

```shell
# 需要在项目根目录来提交, 否则 taiji_client 可能无法正常找到 submit_jobs 目录
cd hunyuan_multimodal_gen_ar

# 提交任务
taiji_client start -cfg submit_jobs/my_task.json
```

## 代码文档

欢迎阅读 [iwiki 文档](https://iwiki.woa.com/p/4015611751). 如果没有权限, 可以找 jarvizhang 申请并说明使用背景.


## 对于新模态开发

* hymm/models 下实现模型结构
* hymm/data_kits 下实现 dataloader
* hymm/trainers 下新建 Trainer 类并继承 BaseTrainer，需要实现 build_dataloader（加载数据）、build_extra_model（加载额外的模型，可选）、prepare_model_inputs（将加载的数据处理为模型输入格式），就可以开始训练了，如有需求可以重写 train_step（计算loss）；如果想边训边测还需要实现 get_sampler 和 eval_step。

## 代码规范

* 安装 black, pip(3) install black
* 对于新增的 python 文件，使用 `black -l 120 new_file.py` 进行格式化，以尽量符合 pep8 规范

## 合入规范

1. 切换到 dev 分支, 拉取最新代码并同步子模块: `git checkout dev && git pull origin dev && git submodule update --init --recursive`
2. 切换到开发分支, rebase 到 dev: `git checkout -b my_feature_branch && git rebase dev`
3. 一个MR内，待合入分支尽量将commit squash成不超过5个 
4. 当前内置了最基础的CI，检查import是否出错，合入前要求CI pass
    * **也欢迎大家补充test_ci下的测试用例，当前CI用例使用的机器是stream流水线自带的，没有CUDA环境，因此CI只能CPU运行，不能使用cuda**
    * 当前CI使用的是基于ptm_flux_v0.16安装了所有依赖的版本，如果新合并的代码引入了更多依赖包，则：1）自行更新docker目录下的Dockerfile并build和push镜像（重命名新的tag）；2）更新.ci/mr_dev.yml配置为新镜像，否则可能失败
