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
    audio = torch.zeros(2, 4800)

    path = _write_video_with_audio(frames, fps=8, audio=audio, audio_sample_rate=48000)
    try:
        with av.open(path) as container:
            assert len(container.streams.video) == 1
            assert len(container.streams.audio) == 1
    finally:
        os.unlink(path)
