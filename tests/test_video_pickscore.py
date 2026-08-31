"""VideoPickScore temporal sampling tests."""

import torch

from unirl.reward.local.pickscore import PickScoreRewardScorer
from unirl.reward.local.video_pickscore import VideoPickScoreScorer
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.reward import RewardRequest


def test_uniform_video_pickscore_averages_sampled_frames(monkeypatch) -> None:
    def fake_scores(self, request):
        del self
        images = request.generated["image"].to_list()
        return [float(image.pixels.mean()) for image in images]

    monkeypatch.setattr(PickScoreRewardScorer, "_compute_model_rewards", fake_scores)
    scorer = object.__new__(VideoPickScoreScorer)
    scorer.frame_selection = "uniform"
    scorer.num_score_frames = 3
    first = torch.stack([torch.full((3, 2, 2), value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)])
    second = torch.stack([torch.full((3, 2, 2), value) for value in (1.0, 0.75, 0.5, 0.25, 0.0)])
    request = RewardRequest(
        primitives={"text": Texts(texts=["up", "down"])},
        generated={"video": Videos.from_list([Video(frames=first), Video(frames=second)])},
    )

    rewards = scorer._compute_model_rewards(request)

    torch.testing.assert_close(torch.tensor(rewards), torch.tensor([0.5, 0.5]), atol=1e-3, rtol=0)
