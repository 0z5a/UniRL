import os
import threading

import numpy as np
import pytest
import torch

from unirl.utils.wandb_logger import UniRLWandBLogger, _write_video_with_audio


def test_media_encoding_runs_off_the_training_thread() -> None:
    logger = UniRLWandBLogger(
        project=None,
        enabled=True,
        log_media=True,
    )
    entered = threading.Event()
    release = threading.Event()

    def encode(*args, **kwargs) -> None:
        entered.set()
        assert release.wait(timeout=5)

    logger._log_generated_media_sync = encode
    logger.log_generated_media(1, {"videos": [torch.zeros(3, 1, 4, 4)]})

    assert entered.wait(timeout=5)
    assert not release.is_set()
    release.set()
    logger.finish()


def test_pyav_mux_writes_video_and_audio_streams() -> None:
    av = pytest.importorskip("av")
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    time = torch.arange(4800, dtype=torch.float32) / 48000
    audio = torch.stack(
        (
            0.5 * torch.sin(2 * torch.pi * 440 * time),
            torch.zeros_like(time),
        ),
        dim=1,
    )

    path = _write_video_with_audio(frames, fps=8, audio=audio, audio_sample_rate=48000)
    try:
        with av.open(path) as container:
            assert len(container.streams.video) == 1
            assert len(container.streams.audio) == 1
            decoded = [frame.to_ndarray() for frame in container.decode(audio=0)]
            waveform = np.concatenate(decoded, axis=-1)
            assert np.mean(waveform[0].astype(float) ** 2) > 0.05
            assert np.mean(waveform[1].astype(float) ** 2) < 1e-6
    finally:
        os.unlink(path)
