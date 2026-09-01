import atexit
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch


def register_failure_state_hooks() -> None:
    if 'LOCAL_RANK' not in os.environ:
        return
    output_dir = "/tmp/torchrun"
    dump_stem = f"R{os.environ.get('LOCAL_RANK', 0)}_pid{os.getpid()}_{datetime.now().isoformat()}"
    status_path = f"{output_dir}/{dump_stem}.status"
    snapshot_path = f"{output_dir}/{dump_stem}.pickle"

    if torch.cuda.is_available():
        from hy_parallelism.tools.profiling import MEMORY_SNAPSHOT_MAX_ENTRIES
        torch.cuda.memory._record_memory_history(
            max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES,
            stacks='python',
        )

    snapshot_dumped = False

    def _record(tag: str, detail: str = "", save_snapshot: bool = False) -> None:
        nonlocal snapshot_dumped
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(status_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().isoformat()} pid={os.getpid()} tag={tag} detail={detail}\n"
                )
                if torch.cuda.is_available():
                    current_device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
                    # torch.cuda.synchronize(current_device)
                    f.write(f'allocated={torch.cuda.memory_allocated(device=current_device) / 1024**3:.2f}G | reserved={torch.cuda.memory_reserved(device=current_device) / 1024**3:.2f}G | peak={torch.cuda.memory_stats(device=current_device)["allocated_bytes.all.peak"] / 1024**3:.2f}G\n')
                    f.write(f"{torch.cuda.memory_stats(device=current_device)}\n")
                    f.write(f"{torch.cuda.memory_summary(device=current_device)}\n")
                    try:
                        gpustat_ret = subprocess.run(
                            ["gpustat"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        f.write("\n--- gpustat ---\n")
                        f.write(gpustat_ret.stdout or "")
                        f.write(gpustat_ret.stderr or "")
                    except Exception as e:
                        f.write(f"\n--- gpustat failed ---\n{e}\n")
                    f.write(f"snapshot: {snapshot_path}\n")
                    if not snapshot_dumped and save_snapshot:
                        snapshot = torch.cuda.memory._snapshot(device=current_device)
                        with open(snapshot_path, "wb") as snapshot_file:
                            pickle.dump(snapshot, snapshot_file)
                        snapshot_dumped = True
                    torch.cuda.synchronize(current_device)
                f.write(
                    f"========================================\n"
                )
        except Exception:
            pass

    def _excepthook(exc_type, exc_value, exc_traceback):
        stack = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _record("unhandled_exception", stack, save_snapshot=True)
        _old_excepthook(exc_type, exc_value, exc_traceback)

    def _signal_handler(signum, frame):
        stack = "".join(traceback.format_stack(frame)) if frame is not None else ""
        _record(f"signal:{signum}", stack, save_snapshot=True)
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    _old_excepthook = sys.excepthook
    sys.excepthook = _excepthook
    atexit.register(lambda: _record("atexit", save_snapshot=True))

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass

    def _heartbeat_loop():
        main_tid = threading.main_thread().ident
        while True:
            time.sleep(60)
            try:
                frame = sys._current_frames().get(main_tid)
                stack = "".join(traceback.format_stack(frame)) if frame is not None else ""
                del frame  # 有可能这个 frame 会保留中间的引用，导致内存不释放
                _record("heartbeat", stack)
            except Exception:
                pass

    threading.Thread(target=_heartbeat_loop, daemon=True).start()
