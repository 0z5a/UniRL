import atexit
import faulthandler
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from hy_parallelism.common.logging import set_trace_log_path


def timeout_command(command: str, timeout: int | float) -> str:
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            status_path = temp_file.name
        os.system(f"timeout {timeout} {command} > {status_path} 2>&1")
        ret = Path(status_path).read_text()
        Path(status_path).unlink()
        return ret
    except Exception as e:
        return f"timeout command failed: {e}"


def register_failure_state_hooks() -> None:
    if 'LOCAL_RANK' not in os.environ:
        return
    output_dir = "/tmp/torchrun"
    dump_stem = f"R{os.environ.get('LOCAL_RANK', 0)}_pid{os.getpid()}_{datetime.now().isoformat()}"
    status_path = f"{output_dir}/{dump_stem}.status"
    snapshot_path = f"{output_dir}/{dump_stem}.pickle"
    set_trace_log_path(f"{output_dir}/{dump_stem}.log")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        # Popen 会让 warning.warn version 变化，dedup 失效
        os.system(f"ps aux | grep -E 'python'  > {status_path}_temp 2>/dev/null")
        process_list = Path(f"{status_path}_temp").read_text()
        Path(f"{status_path}_temp").unlink()
    except Exception as e:
        process_list = f"process_list failed: {e}"

    with open(status_path, "w", encoding="utf-8") as f:
        f.write(f'{os.getcwd()=}\n')
        f.write(f'{sys.argv=}\n')
        f.write(f'{process_list=}\n')
        for key, value in os.environ.items():
            f.write(f"{key}={value}\n")

    if torch.cuda.is_available():
        from hy_parallelism.tools.profiling import start_memory_snapshot
        start_memory_snapshot()

    snapshot_dumped = False

    def _record(tag: str, detail: str = "", save_snapshot: bool = False) -> None:
        nonlocal snapshot_dumped
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # try:
            #     # Popen 会让 warning.warn version 变化，dedup 失效
            #     os.system(f"timeout 3 gpustat > {status_path}_temp 2>/dev/null")
            #     gpustat_text = Path(f"{status_path}_temp").read_text()
            #     Path(f"{status_path}_temp").unlink()
            # except Exception as e:
            #     gpustat_text = f"gpustat failed: {e}"


            try:
                from gpustat.core import GPUStatCollection
                from io import StringIO
                stats = GPUStatCollection.new_query()
                gpustat_text = '\n'.join([k.print_to(StringIO(), with_colors=False, show_pid=True).getvalue() for k in stats.gpus])
            except Exception as e:
                gpustat_text = f"gpustat failed: {e}"

            with open(status_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().isoformat()} pid={os.getpid()} tag={tag}\n"
                )
                if torch.cuda.is_available():
                    current_device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
                    # torch.cuda.synchronize(current_device)
                    allowed = (
                        torch.cuda.get_device_properties(current_device).total_memory
                        * torch.cuda.get_per_process_memory_fraction(current_device)
                        / 1024**3
                    )
                    f.write(f'allocated={torch.cuda.memory_allocated(device=current_device) / 1024**3:.2f}G | reserved={torch.cuda.memory_reserved(device=current_device) / 1024**3:.2f}G | peak={torch.cuda.memory_stats(device=current_device)["allocated_bytes.all.peak"] / 1024**3:.2f}G | allowed={allowed:.2f}G\n')
                    f.write(f"{torch.cuda.memory_stats(device=current_device)}\n")
                    # f.write(f"{torch.cuda.memory_summary(device=current_device)}\n")
                    f.write(f"gpustat: {gpustat_text}\n")
                    f.write(f"snapshot: {snapshot_path}\n")
                    if not snapshot_dumped and save_snapshot:
                        snapshot = torch.cuda.memory._snapshot(device=current_device)
                        with open(snapshot_path, "wb") as snapshot_file:
                            pickle.dump(snapshot, snapshot_file)
                        snapshot_dumped = True
                    # torch.cuda.synchronize(current_device)
                f.write(f"[{tag}] detail: {detail}\n")
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
        # main_tid = threading.main_thread().ident
        while True:
            time.sleep(60)
            try:
                # Holding main-thread frame objects from sys._current_frames() pins
                # f_locals (including large CUDA tensors) until del, which can race
                # with the next diffusion step and cause GPU memory to accumulate.
                # frame = sys._current_frames().get(main_tid)
                # stack = "".join(traceback.format_stack(frame)) if frame is not None else ""
                # del frame  # 有可能这个 frame 会保留中间的引用，导致内存不释放
                # Use faulthandler (C-level dump) so this thread never holds Python
                # frame / local-variable references of other threads.
                # faulthandler requires a real fd (fileno); StringIO is not supported.
                with tempfile.TemporaryFile() as tmp:
                    faulthandler.dump_traceback(file=tmp, all_threads=True)
                    tmp.seek(0)
                    stack = tmp.read().decode("utf-8", errors="backslashreplace")
                _record("heartbeat", stack)
            except Exception:
                pass

    threading.Thread(target=_heartbeat_loop, daemon=True).start()
