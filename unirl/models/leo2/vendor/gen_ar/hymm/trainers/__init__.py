import importlib


class TrainerRegistry(object):
    def __init__(self):
        self._registry = {
            "Text2ImageARTrainer": "text2image_ar_trainer.Text2ImageARTrainer",
            "Text2ImageARTrainer2": "text2image_ar_trainer_2.Text2ImageARTrainer2",
            "InstructionTuningARTrainer": "instruction_tuning_ar_trainer.InstructionTuningARTrainer",
            "Text2ImageTransfusionTrainer": "text2image_transfusion_trainer.Text2ImageTransfusionTrainer",
            "InstructionTuningTransfusionTrainer": "instruction_tuning_transfusion_trainer.InstructionTuningTransfusionTrainer",
            "InstructionTuningTransfusionTrainer2": "instruction_tuning_transfusion_trainer.InstructionTuningTransfusionTrainer2",
            "InstructionTuningGeminiAlphaTrainer": "instruction_tuning_gemini_alpha_trainer.InstructionTuningGeminiAlphaTrainer",
            "Label2ImageTrainerV2": "label2image_trainer.Label2ImageTrainerV2",       # Training with class(e.g., imagenet) as condition
            "Text2ImageMARTrainer": "text2image_mar_trainer.Text2ImageMARTrainer",       # Training with text as condition, using Masked Auto-Regression
            "Text2ImageMARTETrainer": "text2image_mar_te_trainer.Text2ImageMARTETrainer",       # Training with text as condition, using Masked Auto-Regression
            "Text2ImageMLMTrainer": "text2image_mlm_trainer.Text2ImageMLMTrainer",       # Training with text as condition, using Masked Language Model
            "InstructionTuningMLMTrainer": "instruction_tuning_mlm_trainer.InstructionTuningMLMTrainer",
            "Token2ImageDiffusionTrainer": "token2image_diffusion_trainer.Token2ImageDiffusionTrainer",
            "ClipToken2ImageDiffusionTrainer": "cliptoken2image_diffusion_trainer.ClipToken2ImageDiffusionTrainer",
            "VisualText2Audio": "textvisual2audio_trainer.VisualText2Audio",
            "VisualText2AudioMultiHeadTrainer": "textvisual2audio_multihead_trainer.VisualText2AudioMultiHeadTrainer",
            "VisualText2AudioDiffTrainer": "textvisual2audio_diffusion_trainer.VisualText2AudioDiffTrainer",
            "Text2imageText2textTrainer": "text2image_text2text_ar_trainer.Text2imageText2textTrainer",
            "Text2ImageJanusCoTTrainer": "text2image_janus_cot_trainer.Text2ImageJanusCoTTrainer",
            "Text2ImageARGRPOTrainer": "text2image_ar_grpo_trainer.Text2ImageARGRPOTrainer",
        }

    def __getitem__(self, key):
        if key in self._registry:
            key = self._registry[key]

        assert '.' in key, (
            f"Invalid trainer name: {key}. A valid trainer name should be in the form of "
            f"<module_name>.<trainer_cls>."
        )
        module_name, trainer_cls = key.rsplit('.', 1)
        module_spec = importlib.import_module(f"hymm.trainers.{module_name}")
        return getattr(module_spec, trainer_cls)


_registry = TrainerRegistry()


def get_trainer(args):
    return _registry[args.trainer_name](args)
