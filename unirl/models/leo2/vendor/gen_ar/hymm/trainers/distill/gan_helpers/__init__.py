# GAN 蒸馏辅助模块（视音频联合判别器 / 离散时间 shift / teacher 特征抽取）。
#
# 这是一个全新、自包含的子包，迁移自 HunyuanVideo_pureTorch 的 T2VA GAN，
# 并按本仓库（hunyuan_multimoda_gen_ar）的约定改写：
#   - 离散时间步与 shift 严格对齐推理用的 FlowMatchDiscreteScheduler，
#     且 video / audio 各用自己的 flow_shift（gan_time_utils）。
#   - 判别器只有可训练 head（联合视频+音频打分），backbone 复用冻结 teacher
#     （joint_discriminator）。
#   - teacher 中间特征通过 forward hook 非侵入地抽取（feature_extractor），
#     不修改任何现有模型文件。
from .gan_time_utils import (
    GanTimeConfig,
    build_modality_axis,
    sample_shared_step_index,
    gather_times_for_index,
)
from .joint_discriminator import (
    JointDiscriminatorHead,
    get_gan_loss,
)
from .feature_extractor import TeacherFeatureExtractor

__all__ = [
    "GanTimeConfig",
    "build_modality_axis",
    "sample_shared_step_index",
    "gather_times_for_index",
    "JointDiscriminatorHead",
    "get_gan_loss",
    "TeacherFeatureExtractor",
]
