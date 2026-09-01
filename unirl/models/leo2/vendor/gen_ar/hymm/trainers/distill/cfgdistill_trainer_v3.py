# CFG 蒸馏 trainer v3：在 v2 基础上把【音频分支】对称地纳入 CFG 蒸馏。
#
# v2 -> v3
# --------
# 背景（v2 的问题）：
#   模型 forward（leo.py:2066-2073 / 2144-2148）把两支预测放在 **两个独立字段**：
#       LeoOutput.diffusion_prediction        # 视频/图像分支
#       LeoOutput.audio_diffusion_prediction  # 音频分支
#   而 v2（沿用 v1）在 `prepare_cfgdistill_target` 与 `loss_closure` 里 **只取 diffusion_prediction**
#   （视频）。`audio_diffusion_prediction` 从未参与 CFG target 合成、也没有任何 loss，
#   因此音频分支在 CFG 蒸馏中 **完全不被训练**——这正是“视频蒸馏成功、音频偏弱”的根因。
#
# v3 的改动（仅本文件，未改任何既有文件）：
#   1. 完整复用 v2 的负面 prompt 构造（`_build_uncond_tokens` / `prepare_model_inputs`），
#      cond / uncond 仍只在“文本条件”上有差异，video / audio 共享同一份噪声 latent / timestep。
#   2. teacher 的 **同两次** 前向里同时取出 video / audio 的 cond & uncond 预测，
#      分别用 `_calc_cfg_target` 合成两支 CFG target（共用同一个采样得到的 guidance_scale，
#      对应 sampling 的单一 `--diff-guidance-scale`）。不增加额外的 teacher 前向。
#   3. student 同时取 video / audio 预测，分别算 `_cfgdistill_mse`，再按
#      **与普通 flow matching 完全一致** 的权重线性相加：
#            loss = visual_loss_weight * video_mse + audio_loss_weight * audio_mse
#      其中 `visual_loss_weight` / `audio_loss_weight` 直接取自 `prepare_model_inputs`
#      注入 `model_input_kwargs` 的同名字段（来源 args.image_loss_weight=1.0 /
#      args.audio_loss_weight=3.0），与 leo.py 训练分支
#      （loss = visual_loss_weight*visual_diff_loss + audio_loss_weight*audio_diff_loss，
#       L2096 / L2125）**同源同值**，从而保证两支权重与 flow matching 完全一致。
#
import torch

from .cfgdistill_trainer_v2 import MultimodalTrainer as CfgDistillTrainerV2


class MultimodalTrainer(CfgDistillTrainerV2):
    """在 v2（视频-only CFG 蒸馏）之上补齐音频分支的 CFG 蒸馏。"""

    @staticmethod
    def _get_pred(model_output, key):
        """安全地从 LeoOutput(ModelOutput) 取分支预测；字段为 None 时返回 None。

        ModelOutput 会把 None 字段从内部 dict 移除（`output[key]` 可能 KeyError），
        但 dataclass 属性仍存在，故用 getattr 兜底。
        """
        val = getattr(model_output, key, None)
        if val is None:
            try:
                val = model_output[key]
            except (KeyError, TypeError):
                val = None
        return val

    def after_initialize(self):
        super().after_initialize()
        # 注册 video / audio 两支的 CFG 蒸馏分项 loss，便于训练日志监控两支是否都在下降。
        # （loss_dict 里的额外 key 会被 scalar_state 累加，但只有在 loss_names 中才会被 all_reduce 上报。）
        for name in ("video_cfgd_loss", "audio_cfgd_loss"):
            if name not in self.loss_names:
                self.loss_names.append(name)

    @torch.no_grad()
    def prepare_cfgdistill_target(self, model_input_kwargs, uncond_model_input_kwargs, device):
        """同时合成 video / audio 两支的 CFG target。

        Returns
        -------
        video_target : Tensor | list
        audio_target : Tensor | list | None   # 非 av 样本（无音频）时为 None
        sampled_guidance_scale : Tensor (标量)
        """
        guidance_scale_min = getattr(self.args, "guidance_scale_min", 6.0)
        guidance_scale_max = getattr(self.args, "guidance_scale_max", 6.0)
        sampled_guidance_scale = torch.empty((), device=device).uniform_(
            guidance_scale_min, guidance_scale_max
        )

        teacher_model_input_kwargs = model_input_kwargs.copy()
        for k in ("guidance_index", "guidance"):
            teacher_model_input_kwargs.pop(k, None)
        teacher_model_uncond_kwargs = dict(teacher_model_input_kwargs)
        teacher_model_uncond_kwargs.update(uncond_model_input_kwargs)

        # 各一次前向；两支预测都在同一个 LeoOutput 里，不额外增加 teacher 前向次数。
        out_cond = self.teacher_engine(**teacher_model_input_kwargs)
        out_uncond = self.teacher_engine(**teacher_model_uncond_kwargs)

        v_pred_cond = self._get_pred(out_cond, "diffusion_prediction")
        v_pred_uncond = self._get_pred(out_uncond, "diffusion_prediction")
        a_pred_cond = self._get_pred(out_cond, "audio_diffusion_prediction")
        a_pred_uncond = self._get_pred(out_uncond, "audio_diffusion_prediction")

        video_target = self._calc_cfg_target(v_pred_cond, v_pred_uncond, sampled_guidance_scale)
        audio_target = None
        if a_pred_cond is not None and a_pred_uncond is not None:
            audio_target = self._calc_cfg_target(a_pred_cond, a_pred_uncond, sampled_guidance_scale)

        # ===== [CFG-DEBUG] 验证两支 CFG target 是否退化（rel_diff≈0 即负面 prompt 对该支无效）=====
        if getattr(self, "_cfgdbg_pred_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._cfgdbg_pred_left = getattr(self, "_cfgdbg_pred_left", 3) - 1

            def _flatten(x, out):
                if isinstance(x, torch.Tensor):
                    out.append(x.detach().float().reshape(-1))
                elif isinstance(x, (list, tuple)):
                    for xi in x:
                        _flatten(xi, out)

            def _report(tag, pc_raw, pu_raw):
                cp, up = [], []
                _flatten(pc_raw, cp)
                _flatten(pu_raw, up)
                if cp and len(cp) == len(up):
                    pc = torch.cat(cp)
                    pu = torch.cat(up)
                    diff = (pc - pu).abs()
                    self.logger.info(
                        f"[CFG-DEBUG][{tag}] w={sampled_guidance_scale.item():.3f} | "
                        f"numel={pc.numel()} | mean|pred_cond|={pc.abs().mean().item():.4e} | "
                        f"mean|cond-uncond|={diff.mean().item():.4e} | "
                        f"rel_diff={diff.mean().item() / (pc.abs().mean().item() + 1e-8):.4e} "
                        f"(rel_diff≈0 -> 该支 CFG target 退化)"
                    )
                else:
                    self.logger.info(f"[CFG-DEBUG][{tag}] pred 为空或结构无法对齐")

            _report("video", v_pred_cond, v_pred_uncond)
            if a_pred_cond is not None:
                _report("audio", a_pred_cond, a_pred_uncond)
            else:
                self.logger.info("[CFG-DEBUG][audio] 本 batch 无音频分支预测（非 av 样本）")

        del out_cond, out_uncond, v_pred_cond, v_pred_uncond, a_pred_cond, a_pred_uncond
        torch.cuda.empty_cache()
        return video_target, audio_target, sampled_guidance_scale

    def train_step(self, batch):
        args = self.args
        device = torch.device("cuda", args.local_rank)

        model_input_kwargs, uncond_model_input_kwargs, bsz, seqlen = self.prepare_model_inputs(batch, device)

        video_target, audio_target, sampled_guidance_scale = self.prepare_cfgdistill_target(
            model_input_kwargs, uncond_model_input_kwargs, device
        )

        # 与普通 flow matching 完全一致的分支权重：直接复用 prepare_model_inputs 注入的同名字段
        # （来源 args.image_loss_weight / args.audio_loss_weight；非 av 样本时 audio 权重为 0）。
        visual_loss_weight = float(
            model_input_kwargs.get("visual_loss_weight", getattr(args, "image_loss_weight", 1.0))
        )
        audio_loss_weight = float(
            model_input_kwargs.get("audio_loss_weight", getattr(args, "audio_loss_weight", 0.0))
        )

        def loss_closure(model_output, input_args_kwargs_dict):     # noqa
            loss_dict = {}

            # -- 视频分支
            v_pred = self._get_pred(model_output, "diffusion_prediction")
            video_loss = self._cfgdistill_mse(v_pred, video_target)
            total_loss = visual_loss_weight * video_loss
            loss_dict["video_cfgd_loss"] = video_loss.detach()

            # -- 音频分支（仅当本 batch 有音频且权重 > 0 时计入，权重与 flow matching 一致）
            a_pred = self._get_pred(model_output, "audio_diffusion_prediction")
            if audio_target is not None and a_pred is not None and audio_loss_weight > 0:
                audio_loss = self._cfgdistill_mse(a_pred, audio_target)
                total_loss = total_loss + audio_loss_weight * audio_loss
                loss_dict["audio_cfgd_loss"] = audio_loss.detach()

            loss_dict["loss"] = total_loss
            return total_loss, loss_dict

        self.model_engine.register_loss_closure(loss_closure)

        loss = self.model_engine(**model_input_kwargs)

        if torch.isnan(loss).any():
            self.nan_grad_count += 1
            self.logger.warning(
                f"NaN loss encountered in rank {self.rank}, total NaN count: {self.nan_grad_count}"
            )

        loss_dict = self.model_engine.get_cached_result("loss_dict")
        loss_dict['loss'] = loss

        consumed_metrics = {
            batch["dataset_tag"][0]: {
                "samples": batch["n_samples"].sum().item(),
                "tokens": bsz * seqlen
            }
        }

        return loss_dict, consumed_metrics
