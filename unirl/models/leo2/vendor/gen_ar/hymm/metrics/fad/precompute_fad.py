import glob
import os
import numpy as np
import torch
import torchaudio
from .vggish import VGGish, vggish_params
from hymm.constants import VGGISH_MODEL_PATH, VGGISH_PCA_PATH
from loguru import logger

@torch.no_grad()
def cal_embedding_stats(audio_list):
    results = []
    for idx, audio_file in enumerate(audio_list):
        logger.info(f"{idx}/{len(audio_list)}: {audio_file}")
        audio_sample, sample_rate = torchaudio.load(audio_file)
        audio_sample = audio_sample.squeeze(0)
        if sample_rate != vggish_params.SAMPLE_RATE:
            audio_sample = audio_sample
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=vggish_params.SAMPLE_RATE)
            audio_sample = resampler(audio_sample)
        audio_sample = audio_sample.cuda()
        logger.info(audio_sample.shape)
        audio_feats = vggish(audio_sample)
        logger.info(audio_feats.shape)
        results.append(audio_feats.cpu().numpy())
    
    predictions = np.concatenate(results, axis=0)
    logger.info(predictions.shape)
    mu, sigma = np.mean(predictions, axis=0), np.cov(predictions, rowvar=False)
    logger.info(mu.shape)
    logger.info(sigma.shape)
    return {"mu": mu, "sigma": sigma}


if __name__ == '__main__':
    audio_folder = "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets/evaluation/dataset/audioset/val/audios"
    output_pt_path = "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets/evaluation/FAD/audioset_vggish_fad_stats.pt"

    vggish = VGGish(VGGISH_MODEL_PATH, VGGISH_PCA_PATH).cuda().eval()
    logger.info(vggish)

    audio_list = sorted(glob.glob(os.path.join(audio_folder, "*.wav")))
    res = cal_embedding_stats(audio_list)
    torch.save(res, output_pt_path)
