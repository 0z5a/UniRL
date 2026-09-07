from types import SimpleNamespace

import torch

from unirl.distributed.tensor.ref import TensorSpan
from unirl.distributed.tensor.worker_local import WorkerLocalTransport


class _SourceHandle:
    def __init__(self, key: str) -> None:
        self.store_key = key
        self.shape = (2, 3)
        self.dtype = torch.float32
        self.device = "cuda"


class _ReceivedHandle(_SourceHandle):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.bound_worker = None

    def rebind(self, worker) -> None:
        self.bound_worker = worker


class _RemoteMethod:
    def __init__(self, device_id: int, calls: list) -> None:
        self.device_id = device_id
        self.calls = calls

    def remote(self, op, *args):
        call = (self.device_id, op, args)
        self.calls.append(call)
        return call


class _Pool:
    def __init__(self, calls: list) -> None:
        self.workers = {
            device_id: SimpleNamespace(
                transport_op=_RemoteMethod(device_id, calls),
            )
            for device_id in range(6)
        }

    def slot0_worker(self, device_id: int):
        return self.workers[device_id]


def test_worker_local_move_launches_disjoint_device_pairs_in_one_wave(monkeypatch) -> None:
    calls = []
    get_batch_sizes = []
    pool = _Pool(calls)

    def fake_get(refs):
        get_batch_sizes.append(len(refs))
        results = []
        for device_id, op, args in refs:
            if op == "nccl_recv":
                shapes = args[1]
                results.append(
                    [_ReceivedHandle(f"received-{device_id}-{index}") for index in range(len(shapes))]
                )
            else:
                results.append(None)
        return results

    monkeypatch.setattr("unirl.distributed.tensor.worker_local.ray.get", fake_get)
    spans = {
        (0, 1, "a", 0, 2): TensorSpan(_SourceHandle("a"), 0, 2),
        (2, 3, "b", 0, 2): TensorSpan(_SourceHandle("b"), 0, 2),
    }

    moved = WorkerLocalTransport._move(pool, spans)

    assert get_batch_sizes == [4]
    assert set(moved) == set(spans)
    assert all(span.handle.bound_worker is pool.slot0_worker(key[1]) for key, span in moved.items())


def test_worker_local_move_serializes_pairs_that_share_a_device(monkeypatch) -> None:
    calls = []
    get_batch_sizes = []
    pool = _Pool(calls)

    def fake_get(refs):
        get_batch_sizes.append(len(refs))
        results = []
        for device_id, op, args in refs:
            if op == "nccl_recv":
                results.append([_ReceivedHandle(f"received-{device_id}") for _ in args[1]])
            else:
                results.append(None)
        return results

    monkeypatch.setattr("unirl.distributed.tensor.worker_local.ray.get", fake_get)
    spans = {
        (0, 1, "a", 0, 2): TensorSpan(_SourceHandle("a"), 0, 2),
        (0, 2, "b", 0, 2): TensorSpan(_SourceHandle("b"), 0, 2),
    }

    WorkerLocalTransport._move(pool, spans)

    assert get_batch_sizes == [2, 2]
