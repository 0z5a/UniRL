from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_vendor_leo(monkeypatch):
    vendor_root = Path(__file__).resolve().parents[1] / "unirl" / "models" / "leo2" / "vendor" / "gen_ar"
    monkeypatch.syspath_prepend(str(vendor_root / "deps" / "hy_parallelism"))
    monkeypatch.syspath_prepend(str(vendor_root))
    return importlib.import_module("hymm.models.diffusion.leo")


@pytest.mark.parametrize("repeats", [[5, 12]])
def test_shared_timestep_uses_broadcast_with_equal_value_and_gradient(
    monkeypatch,
    repeats,
) -> None:
    leo = _load_vendor_leo(monkeypatch)
    # The singleton path must run before CP's index/scatter implementation too.
    monkeypatch.setattr(leo, "get_parallel_state", lambda: SimpleNamespace(cp_size=2))
    timestep_mod = torch.randn(1, 1, 8, requires_grad=True)
    hidden = torch.randn(1, 17, 8)

    broadcast_mod = leo._repeat_interleave(
        timestep_mod,
        torch.tensor(repeats),
        dim=1,
    )
    explicit_mod = timestep_mod.expand(1, hidden.shape[1], -1)

    assert broadcast_mod.shape == (1, 1, 8)
    assert broadcast_mod.data_ptr() == timestep_mod.data_ptr()
    actual = (hidden * broadcast_mod).sum()
    expected = (hidden * explicit_mod).sum()
    torch.testing.assert_close(actual, expected)

    actual.backward(retain_graph=True)
    actual_grad = timestep_mod.grad.detach().clone()
    timestep_mod.grad = None
    expected.backward()
    torch.testing.assert_close(timestep_mod.grad, actual_grad)


def test_single_sample_timestep_uses_native_repeat_interleave(monkeypatch) -> None:
    leo = _load_vendor_leo(monkeypatch)
    monkeypatch.setattr(leo, "get_parallel_state", lambda: SimpleNamespace(cp_size=1))
    timestep_mod = torch.randn(1, 1, 8, requires_grad=True)

    result = leo._repeat_interleave(timestep_mod, torch.tensor([17]), dim=1)

    torch.testing.assert_close(result, torch.repeat_interleave(timestep_mod.float(), torch.tensor([17]), dim=1))
    assert result.shape == (1, 17, 8)


def test_multiple_timesteps_still_materialize_distinct_token_modulation(monkeypatch) -> None:
    leo = _load_vendor_leo(monkeypatch)
    monkeypatch.setattr(leo, "get_parallel_state", lambda: SimpleNamespace(cp_size=1))
    timestep_mod = torch.tensor([[[1.0], [2.0]]])

    result = leo._repeat_interleave(timestep_mod, torch.tensor([2, 3]), dim=1)

    torch.testing.assert_close(result, torch.tensor([[[1.0], [1.0], [2.0], [2.0], [2.0]]]))


def test_timestep_repeat_rejects_ambiguous_cardinality(monkeypatch) -> None:
    leo = _load_vendor_leo(monkeypatch)
    monkeypatch.setattr(leo, "get_parallel_state", lambda: SimpleNamespace(cp_size=1))

    with pytest.raises(ValueError, match="one shared state or one state per sample"):
        leo._repeat_interleave(torch.ones(1, 2, 1), torch.tensor([2, 3, 4]), dim=1)
