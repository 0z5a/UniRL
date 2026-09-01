# This trainer is based on pure torch and extends cfgdistill_trainer_v1.
#
# v1 vs v2
# --------
# - v1 是验证管线：`prepare_model_inputs` 的 cond / uncond 返回的是 **同一份** kwargs
#   （占位），因此 teacher 的 pred_cond == pred_uncond，CFG target 退化为 pred_cond，
#   只能用来跑通流程。
# - v2 真正构造了 **无条件（负面）文本条件** `uncond_model_input_kwargs`，使其与采样阶段
#   CFG 所用的 negative prompt **完全一致**：把 prompt 文本 token 逐个替换为可学习的
#   null token `<cfg>`（`uncond_length="equal"`，保持长度不变），bos / 分隔符 / 媒体占位符
#   等结构性 special token 保持不变，再用同一个 text encoder 重新编码。
#
# 这样 teacher 在 **同一份噪声 latent / timestep** 上分别给出 cond / uncond 的 velocity，
# 二者按 CFG 公式合成：
#       target = pred_uncond + w * (pred_cond - pred_uncond)
# 该 target 被蒸馏进 student（self.model）。共享 latent / timestep 是合法 CFG target 的前提，
# 因此 v2 只覆盖文本条件项，其余（latents / timesteps / attention_mask / rope_media_info ...）
# 全部复用 cond。
#
import torch

from ...core.data_provider_dit import (
    encode_text,
    prepare_model_inputs,
)
from ...core.global_vars import get_tkwrapper
from .cfgdistill_trainer_v1 import MultimodalTrainer as CfgDistillTrainerV1


class MultimodalTrainer(CfgDistillTrainerV1):
    """CFG 蒸馏 trainer v2：构造与采样一致的负面（无条件）文本条件。

    复用 v1 的全部能力（teacher 构建、`_calc_cfg_target`、`_cfgdistill_mse`、
    `prepare_cfgdistill_target`、`train_step`、`validation` 等），仅重写
    `prepare_model_inputs`，使其在返回 cond kwargs 的同时，额外构造真正的
    `uncond_model_input_kwargs`。
    """

    def _build_uncond_tokens(self, tokens, text_mask, device):
        """在 token 级别复刻采样阶段 CFG 的负面 prompt。

        采样（`uncond_length="equal"`）的做法：把每个 **prompt 文本 token** 替换为可学习的
        null token ``<cfg>``，序列长度保持不变；而 bos / 分隔符 / 媒体占位符等结构性
        special token 不动。这里完全复现该行为。

        Parameters
        ----------
        tokens : LongTensor [bsz, seqlen]
            打包后的完整多模态 token 序列（文本 + 媒体占位符），来自 ``batch["tokens"]``。
        text_mask : Tensor [bsz, seqlen]
            文本位掩码：有效文本=1，媒体 / padding=0。在本代码库中它与 ``tokens`` 已对齐
            （见 ``data_provider_dit.encode_text`` 直接消费 ``batch["text_mask"]``，无需移位）。
        """
        # tkwrapper 在训练初始化时由 build_tkwrapper() 设为全局；优先用 trainer 上的引用。
        tkwrapper = getattr(self, "tkwrapper", None) or get_tkwrapper()
        uncond_token_id = getattr(tkwrapper, "cfg_token_id", None)
        if uncond_token_id is None:
            raise ValueError("Cannot build uncond tokens: cfg_token_id is unavailable.")

        tokens = tokens.to(device)
        text_mask_bool = text_mask.to(device).to(torch.bool)

        # 防御性对齐：理论上二者等长，此处与团队参考实现保持一致以兜底极端情况。
        if text_mask_bool.shape[1] < tokens.shape[1]:
            pad = torch.zeros(
                (tokens.shape[0], tokens.shape[1] - text_mask_bool.shape[1]),
                dtype=torch.bool, device=device,
            )
            text_mask_bool = torch.cat([text_mask_bool, pad], dim=1)
        elif text_mask_bool.shape[1] > tokens.shape[1]:
            text_mask_bool = text_mask_bool[:, :tokens.shape[1]]

        # 仅替换真正的 prompt 内容，绝不覆盖结构性 special token（bos / 分隔符 / <img> ...）。
        # 这与采样完全一致——采样中只有用户 prompt 文本会变成 <cfg>，模板 / 系统提示词不变。
        special_token_ids = [
            v for v in getattr(tkwrapper, "special_token_map", {}).values()
            if isinstance(v, int)
        ]
        if special_token_ids:
            special_ids = torch.tensor(special_token_ids, device=device, dtype=tokens.dtype)
            special_mask = torch.isin(tokens, special_ids)
            replace_mask = text_mask_bool & (~special_mask)
        else:
            replace_mask = text_mask_bool

        uncond_tokens = tokens.clone()
        uncond_tokens[replace_mask] = int(uncond_token_id)

        # ===== [CFG-DEBUG] 验证“负面 prompt”是否真的替换到了 prompt 文本 =====
        # loss 反推：loss = (w-1)^2 * ||pred_cond - pred_uncond||^2，若 loss~1e-4 则 cond≈uncond，
        # 说明 uncond_tokens ≈ tokens（替换没生效）。这里把替换情况量化打印出来。
        if getattr(self, "_cfgdbg_steps_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._cfgdbg_steps_left = getattr(self, "_cfgdbg_steps_left", 3) - 1
            tm_sum = int(text_mask_bool.sum().item())
            sp_sum = int((text_mask_bool & special_mask).sum().item()) if special_token_ids else 0
            rep_sum = int(replace_mask.sum().item())
            changed = int((uncond_tokens != tokens).sum().item())
            self.logger.info(
                f"[CFG-DEBUG] cfg_token_id={int(uncond_token_id)} | tokens={list(tokens.shape)} | "
                f"text_mask=1 count={tm_sum} | special&text count={sp_sum} | "
                f"replace_mask count={rep_sum} | actually_changed={changed}"
            )
            # 抽样看第 0 条样本被替换前后的前若干个文本位置 token
            row0 = text_mask_bool[0].nonzero(as_tuple=True)[0]
            if row0.numel() > 0:
                sl = row0[:20]
                self.logger.info(
                    f"[CFG-DEBUG] sample0 text-pos orig tokens={tokens[0, sl].tolist()}"
                )
                self.logger.info(
                    f"[CFG-DEBUG] sample0 text-pos uncond tokens={uncond_tokens[0, sl].tolist()}"
                )
        return uncond_tokens

    def prepare_model_inputs(self, batch, device):
        # ---- 条件分支：与 v1 完全相同 ----
        # prepare_model_inputs 内部会做 VAE 编码 + 加噪（带 RNG），生成 x_t / timesteps。
        # 这一步只调用一次，cond 与 uncond 将共享同一份噪声 latent / timestep。
        with torch.autocast(device_type="cuda", enabled=False):
            model_input_kwargs, bsz, seqlen = prepare_model_inputs(
                batch, device, model_config=self.model.get_config(),
            )

            # ---- 无条件分支：重新编码“负面 prompt” ----
            # 用同一个 batch，但把 prompt 文本 token 换成 <cfg>，再走同一个 text encoder。
            # 注意：这里只重做文本编码，不再做 VAE / 加噪，故不会引入第二份噪声。
            uncond_tokens = self._build_uncond_tokens(
                batch["tokens"], batch["text_mask"], device,
            )
            uncond_batch = dict(batch)            # 浅拷贝：仅替换 tokens，其余张量共享引用
            uncond_batch["tokens"] = uncond_tokens

            dataset_tag = batch["dataset_tag"][0]
            task_kwargs = getattr(self.args, f"{dataset_tag}_task_kwargs")
            uncond_text_states, uncond_text_mask = encode_text(
                self.text_encoder, uncond_batch, task_kwargs, device,
            )

        # cond 与 uncond 的差异 **仅在文本条件**；latents / timesteps / attention_mask /
        # rope_media_info 等全部复用 cond。prepare_cfgdistill_target() 通过
        #     teacher_uncond = dict(teacher_cond); teacher_uncond.update(uncond_kwargs)
        # 来合成 uncond 输入，所以这里只需提供需要覆盖的文本条件项。
        uncond_model_input_kwargs = {
            "cond_text_states": uncond_text_states,
            "cond_text_mask": uncond_text_mask,
        }

        # ===== [CFG-DEBUG] cond vs uncond 文本编码状态差异 =====
        # 若此差异≈0：token 替换没生效或编码器对 <cfg> 不敏感 -> 负面 prompt 失效。
        if getattr(self, "_cfgdbg_states_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._cfgdbg_states_left = getattr(self, "_cfgdbg_states_left", 3) - 1
            cond_states = model_input_kwargs.get("cond_text_states")
            if cond_states is not None and uncond_text_states is not None \
                    and cond_states.shape == uncond_text_states.shape:
                d = (cond_states.float() - uncond_text_states.float())
                self.logger.info(
                    f"[CFG-DEBUG] text_states shape={list(cond_states.shape)} | "
                    f"mean|cond|={cond_states.float().abs().mean().item():.4e} | "
                    f"mean|cond-uncond|={d.abs().mean().item():.4e} | "
                    f"max|cond-uncond|={d.abs().max().item():.4e}"
                )
            else:
                self.logger.info(
                    f"[CFG-DEBUG] text_states shape mismatch or None: "
                    f"cond={None if cond_states is None else list(cond_states.shape)} "
                    f"uncond={None if uncond_text_states is None else list(uncond_text_states.shape)}"
                )

        return model_input_kwargs, uncond_model_input_kwargs, bsz, seqlen

    @torch.no_grad()
    def prepare_cfgdistill_target(self, model_input_kwargs, uncond_model_input_kwargs, device):
        """覆写仅为加诊断：打印 teacher 的 pred_cond vs pred_uncond 真实差异与采样的 w。
        逻辑与父类 (v1) 完全一致。"""
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

        pred_cond = self.teacher_engine(**teacher_model_input_kwargs)["diffusion_prediction"]
        pred_uncond = self.teacher_engine(**teacher_model_uncond_kwargs)["diffusion_prediction"]

        if getattr(self, "_cfgdbg_pred_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._cfgdbg_pred_left = getattr(self, "_cfgdbg_pred_left", 3) - 1

            # pred 可能是 Tensor，也可能是任意深度嵌套的 list[Tensor]。递归展平成 fp32 一维向量。
            def _flatten(x, out):
                if isinstance(x, torch.Tensor):
                    out.append(x.detach().float().reshape(-1))
                elif isinstance(x, (list, tuple)):
                    for xi in x:
                        _flatten(xi, out)

            cond_parts, uncond_parts = [], []
            _flatten(pred_cond, cond_parts)
            _flatten(pred_uncond, uncond_parts)
            if cond_parts and len(cond_parts) == len(uncond_parts):
                pc = torch.cat(cond_parts)
                pu = torch.cat(uncond_parts)
                diff = (pc - pu).abs()
                self.logger.info(
                    f"[CFG-DEBUG] w={sampled_guidance_scale.item():.3f} | "
                    f"n_tensors={len(cond_parts)} | numel={pc.numel()} | "
                    f"mean|pred_cond|={pc.abs().mean().item():.4e} | "
                    f"mean|pred_cond-pred_uncond|={diff.mean().item():.4e} | "
                    f"rel_diff={diff.mean().item() / (pc.abs().mean().item() + 1e-8):.4e} "
                    f"(若 rel_diff≈0 则 CFG target 退化为 pred_cond -> 负面 prompt 失效)"
                )
            else:
                self.logger.info(
                    f"[CFG-DEBUG] pred 结构无法对齐: cond_parts={len(cond_parts)} "
                    f"uncond_parts={len(uncond_parts)} | "
                    f"type(pred_cond)={type(pred_cond)}"
                )

        cfgdistill_target = self._calc_cfg_target(pred_cond, pred_uncond, sampled_guidance_scale)
        del pred_cond, pred_uncond
        torch.cuda.empty_cache()
        return cfgdistill_target, sampled_guidance_scale
