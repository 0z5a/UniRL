import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import loguru
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pygments import console

class PrecisionAligner:
    """
    A precision alignment tool for comparing outputs from different PyTorch runs with different configurations.
    
    This tool handles cases where different runs may have different world_size, rank, or parallel configurations.
    It saves tensors to a shared directory and compares them across different tasks.
    
    Example usage:
        # First run (e.g., 8 cards, sp1, dp8)
        import torch
        precision_aligner = PrecisionAligner(
            task_name='plan1', 
            tasks=['plan1', 'plan2'], 
            work_dir='./align_outputs',
            input_src='plan1'
        )
        x = torch.randn(3)
        x = precision_aligner.handle_input('input', x)
        y = x * 4
        precision_aligner.align('y', y)

        # Second run (e.g., 8 cards, sp2, dp4)
        precision_aligner = PrecisionAligner(
            task_name='plan2', 
            tasks=['plan1', 'plan2'], 
            work_dir='./align_outputs',
            input_src='plan1'
        )
        x = torch.randn(3)
        x = precision_aligner.handle_input('input', x)
        y = x * 4
        precision_aligner.align('y', y)
    """

    def __init__(
        self,
        task_name: str,
        tasks: List[str],
        work_dir: str,
        input_src: str,
        align_ranks: List[int] = None,
        dp_rank: int = -1,
        is_parallel_zero: bool = True,
        cache_tasks: bool = False,
        enable: bool = True,
    ):
        """
        Initialize the PrecisionAligner.
        
        Args:
            task_name: Name of the current task/run
            tasks: List of all task names to compare
            work_dir: Directory to save tensors for comparison
            input_src: Source task name for input tensors
            align_ranks: List of ranks to perform alignment on (default: [0])
            dp_rank: Data parallel rank (-1 means auto-detect)
            is_parallel_zero: Whether this is a parallel zero configuration
            enable: Whether to enable precision alignment
        """
        assert task_name, "task_name cannot be empty"
        assert tasks, "tasks cannot be empty"
        assert task_name in tasks, f"{task_name} not in {tasks}"
        assert work_dir, "work_dir cannot be empty"
        assert input_src, "input_src cannot be empty"

        self.task_name = task_name
        self.tasks = tasks
        self.work_dir = work_dir
        self.align_ranks = align_ranks if align_ranks is not None else [0]
        self.enable = enable
        self.dp_rank = dp_rank
        self.is_parallel_zero = is_parallel_zero
        self.input_src = input_src
        self.cache_tasks = cache_tasks
        self.cached = []

        if self.enable:
            try:
                loguru.logger.debug(
                    f"Aligning precision for {tasks}, current run: {task_name}. "
                    f"Saving to {work_dir}"
                )
                if dist.get_rank() == 0:
                    self.clear()
                self.set_seed(0)
            except Exception as e:
                print(f"Warning: Could not log alignment info: {e}")

    def get_rank(self) -> int:
        """
        Get the current rank for tensor saving/loading.
        
        Returns:
            Current rank (either dp_rank or auto-detected)
        """
        if self.dp_rank >= 0:
            return self.dp_rank

        # Try to get rank from environment variables
        rank = os.environ.get('RANK', None)
        if rank is not None:
            try:
                return int(rank)
            except ValueError:
                pass

        # Try to get rank from distributed training
        try:
            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass

        # Fallback to 0
        return 0

    def should_run(self) -> bool:
        """Check if the current rank should perform alignment operations."""
        return self.get_rank() in self.align_ranks

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible results."""
        try:
            from accelerate.utils import set_seed

            set_seed(seed)
        except ImportError:
            import random

            import numpy as np

            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def get_input_tensor_path(self, tag: str) -> str:
        """Get the file path for saving/loading input tensors."""
        directory = Path(self.work_dir)
        directory.mkdir(exist_ok=True, parents=True)
        return str(directory / f"input_{tag}_rank{self.get_rank()}.pt")

    def handle_input(self, tag: str, obj: Any, force_not_read: bool = False) -> Any:
        """
        Handle input tensors to ensure consistency across different runs.
        
        Args:
            tag: Input tensor tag
            obj: Input object/tensor
            force_not_read: Force not to read from saved file
            
        Returns:
            Input object (either original or loaded from file)
        """
        if not self.enable:
            return obj

        self.set_seed(0)
        path = self.get_input_tensor_path(tag)

        # Use barrier only if distributed training is initialized
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

        if not Path(path).exists():
            time.sleep(1)
            loguru.logger.debug("No shared input found, saving current input")
            assert self.task_name == self.input_src, (
                f"Current task {self.task_name} must be input source "
                f"{self.input_src}"
            )

            if self.is_parallel_zero:
                try:
                    torch.save(obj, path)
                    loguru.logger.debug(f"Saved input tensor to {path}")
                except Exception as e:
                    loguru.logger.error(f"Failed to save input tensor: {e}")
                    loguru.logger.exception(e)
            return obj
        else:
            if not force_not_read:
                loguru.logger.debug(f"Using shared input: {path}")
                try:
                    return torch.load(
                        path, map_location="cuda", weights_only=False
                    )
                except Exception as e:
                    loguru.logger.error(f"Failed to load input tensor: {e}")
                    return obj
            return obj

    def align(
        self,
        tag: str,
        tensor: Any,
        save_only: bool = False,
        pause: bool = False,
        atol=None,
        rtol=None,
    ) -> None:
        """
        Align and compare tensors across different tasks.

        Args:
            tag: Tag name for the tensor
            tensor: Tensor or object to save/compare
            save_only: Only save, don't compare
            pause: Whether to pause execution on mismatch
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
        """
        if not self.should_run():
            return
        if not self.enable:
            return

        # assert isinstance(tensor, (torch.Tensor, int, float, bool, str)), f'tensor must be a tensor, int, float, bool, str, list, tuple, or dict, but got {type(tensor)}'

        if self.cache_tasks:
            self.cached.append((tag, tensor, save_only, pause))
            return

        curr_path = self.get_tensor_path(self.task_name, tag)

        # Save current tensor
        if self.is_parallel_zero:
            if Path(curr_path).exists():
                loguru.logger.warning(
                    f"{curr_path} already exists, consider if running multiple times or multiple precision checks"
                )
            try:
                loguru.logger.debug(f"Saving tensor to {curr_path}")
                torch.save(tensor, curr_path)
            except Exception as e:
                loguru.logger.error(f"Failed to save tensor to {curr_path}: {e}")
                return

        if save_only:
            return
        if not self.is_parallel_zero:
            return

        # Compare with other tasks
        try:
            exists = all(
                [ Path(self.get_tensor_path(task, tag)).exists() for task in self.tasks ]
            )
            if not exists:
                loguru.logger.warning(f"Not all tasks have completed `{tag}` yet")
                return
            tensors = []
            for task in self.tasks:
                try:
                    task_tensor = torch.load(
                        self.get_tensor_path(task, tag),
                        map_location="cuda",
                        weights_only=False,
                    )
                    tensors.append(task_tensor)
                except Exception as e:
                    loguru.logger.error(
                        f"Failed to load tensor from task {task}: {e}"
                    )
                    return

            # Compare tensors
            diff_results = []
            for i, t in enumerate(tensors):
                result = self.compare(
                    tensor,
                    t,
                    f"{tag} [{self.task_name} rank vs {self.tasks[i]}]",
                    atol=atol,
                    rtol=rtol,
                )
                diff_results.append(result)

            all_close = all(diff_results)
            if not all_close:
                loguru.logger.warning(
                    f"`{tag}` mismatch detected, {self.tasks} | diff vs {self.task_name}"
                )
                if pause:
                    self._pause_execution()
            else:
                loguru.logger.info(
                    f"`{tag}` precision check passed | {self.tasks}"
                )
        except Exception as e:
            loguru.logger.error(f"Comparison failed for `{tag}`: {e}")


    def compare(
        self,
        a: Any,
        b: Any,
        tag: str,
        atol=None,
        rtol=None,
        check_dtype: bool = True,
        detailed_output: bool = True,
    ) -> bool:
        """
        Compare two objects for equality.

        Args:
            a: First object to compare
            b: Second object to compare
            tag: Tag name for logging
            atol: Absolute tolerance for tensor comparison
            rtol: Relative tolerance for tensor comparison
            check_dtype: Whether to check dtype for tensor comparison
            detailed_output: Whether to output detailed error messages

        Returns:
            True if objects are equal, False otherwise
        """
        # Type check
        if type(a) is not type(b):
            error_msg = (
                f"Comparing {tag}: Type mismatch - "
                f"{type(a).__name__} vs {type(b).__name__}"
            )
            if detailed_output:
                loguru.logger.error(error_msg)
            return False

        # PyTorch Tensor comparison
        if isinstance(a, torch.Tensor):
            return self._compare_tensor(
                a, b, tag, atol, rtol, check_dtype, detailed_output
            )

        # Basic type comparison
        try:
            if a != b:
                if detailed_output:
                    loguru.logger.error(
                        f"Comparing {tag}: Value mismatch - {a} != {b}"
                    )
                return False
        except Exception as e:
            raise Exception(f'Comparing {tag}: {type(a)} vs {type(b)} error: {e}')
        return True

    def _compare_tensor(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        tag: str,
        atol=None,
        rtol=None,
        check_dtype: bool = True,
        detailed_output: bool = True,
    ) -> bool:
        """
        Compare two PyTorch tensors for equality.

        Args:
            a: First tensor to compare
            b: Second tensor to compare
            tag: Tag name for logging
            atol: Absolute tolerance
            rtol: Relative tolerance
            check_dtype: Whether to check dtype
            detailed_output: Whether to output detailed error messages

        Returns:
            True if tensors are close, False otherwise
        """
        b = b.to(a.device)
        try:
            torch.testing.assert_close(
                a,
                b,
                atol=atol,
                rtol=rtol,
                check_device=False,
                check_dtype=check_dtype,
                msg=lambda msg: f"Comparing `{tag}`: {msg}",
            )
        except Exception as e:
            if detailed_output:
                loguru.logger.error(f"{e}")
            return False
        return True


    def _pause_execution(self) -> None:
        """Pause execution for debugging."""
        try:
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    import pdb
                    pdb.set_trace()
                torch.distributed.barrier()
            if 'RANK' in os.environ:
                if int(os.environ['RANK']) == 0:
                    import pdb
                    pdb.set_trace()
            else:
                import pdb
                pdb.set_trace()
        except Exception as e:
            loguru.logger.warning(f'Could not pause execution: {e}')

    def get_tensor_path(self, task_name: str, tag: str) -> str:
        """Get the file path for saving/loading a tensor."""
        directory = Path(self.work_dir) / task_name
        directory.mkdir(exist_ok=True, parents=True)
        return str(directory / f"{tag}_rank{self.get_rank()}.pt")

    def clear(self) -> None:
        """Clear saved tensors for the current task."""
        import shutil
        directory = Path(self.work_dir) / self.task_name
        if directory.exists():
            try:
                shutil.rmtree(directory)
                loguru.logger.debug(f"Cleared directory: {directory}")
            except Exception as e:
                loguru.logger.error(f"Failed to clear directory {directory}: {e}")


    def execute_align_tasks(self) -> None:
        """Execute alignment tasks."""
        self.cache_tasks = False
        for task in self.cached:
            print(f'Aligning {task[0]}')
            self.align(*task)
        self.cached = []
        self.cache_tasks = True

_global_aligner = None

def create_aligner(
    task_name: str,
    tasks: List[str],
    work_dir: str,
    input_src: str,
    align_ranks: List[int] = None,
    dp_rank: int = -1,
    is_parallel_zero: bool = True,
    cache_tasks: bool = False,
    enable: bool = True
) -> PrecisionAligner:
    global _global_aligner

    _global_aligner = PrecisionAligner(
        task_name=task_name,
        tasks=tasks,
        work_dir=work_dir,
        input_src=input_src,
        align_ranks=align_ranks,
        dp_rank=dp_rank,
        is_parallel_zero=is_parallel_zero,
        cache_tasks=cache_tasks,
        enable=enable
    )
    return _global_aligner

def get_global_aligner() -> PrecisionAligner:
    assert _global_aligner is not None, "global aligner is not initialized"
    return _global_aligner


# --- Simple tensor / state_dict precision report (no name remapping) ---


@dataclass
class PrecisionReportThresholds:
    """All comparisons are in the sense: align if value is within a good range."""

    # Absolute diff must be at or below this to pass
    max_diff: float = 1e-4
    # Relative error quantiles at 95th / 90th / 80th (ratio |a-b|/|a| where diff != 0); each must be <= bound
    p95: float = 0.01
    p90: float = 0.01
    p80: float = 0.01
    # Cosine similarity must be at or above
    cosine_min: float = 0.9999
    # |sum|a| - sum|b|| / (max(sum|a|, sum|b|) + eps); None = do not use in pass/fail
    sum_rel_diff_max: Optional[float] = None

    def is_aligned(self, row: Dict[str, Any]) -> bool:
        if not row.get("ok_compare", True):
            return False
        md = row["max_diff"]
        p95, p90, p80 = row["p95"], row["p90"], row["p80"]
        if md != md or p90 != p90 or p95 != p95:  # NaN guard
            return False
        if md > self.max_diff:
            return False
        if p95 > self.p95 or p90 > self.p90 or p80 > self.p80:
            return False
        if row["cosine_sim"] < self.cosine_min:
            return False
        if self.sum_rel_diff_max is not None and row.get("sum_rel_diff", 0.0) > self.sum_rel_diff_max:
            return False
        return True

    def failed_metric_table_columns(self, row: Dict[str, Any]) -> Set[str]:
        """
        Which metric column *names* (must match the printed header) are over threshold.
        Only defined when ``ok_compare``; used for target red highlight on the failing value.
        """
        if not row.get("ok_compare", True):
            return set()
        failed: Set[str] = set()
        md, p95, p90, p80 = row["max_diff"], row["p95"], row["p90"], row["p80"]
        if md == md and md > self.max_diff:
            failed.add("max_diff")
        if p95 == p95 and p95 > self.p95:
            failed.add("p95")
        if p90 == p90 and p90 > self.p90:
            failed.add("p90")
        if p80 == p80 and p80 > self.p80:
            failed.add("p80")
        cos = row["cosine_sim"]
        if cos == cos and cos < self.cosine_min:
            failed.add("cosine_sim")
        srd = row.get("sum_rel_diff", 0.0)
        if self.sum_rel_diff_max is not None and srd == srd and srd > self.sum_rel_diff_max:
            failed.add("sum_rel_diff")
        return failed


# Peak working set for one pair: float32 a/b/diff + gathered rel ≈ 4 * numel * 4 bytes.
_COMPARE_BYTES_PER_ELEM = 16
_COMPARE_MEM_FRACTION = 0.7
_QUANTILE_LEVELS = (0.95, 0.90, 0.80)
# torch.quantile (esp. CUDA) rejects inputs larger than ~2^24 elements.
_TORCH_QUANTILE_MAX_ELEMS = 1 << 24


def _rel_error_quantiles(rel: torch.Tensor) -> Tuple[float, float, float]:
    """p95/p90/p80 of relative errors; falls back when torch.quantile size-limits."""
    n = rel.numel()
    if n == 0:
        return 0.0, 0.0, 0.0
    if n <= _TORCH_QUANTILE_MAX_ELEMS:
        try:
            qs = torch.tensor(_QUANTILE_LEVELS, device=rel.device, dtype=rel.dtype)
            return tuple(float(x) for x in torch.quantile(rel, qs).tolist())
        except RuntimeError:
            pass
    rel_np = rel.detach().float().cpu().numpy()
    return tuple(float(np.quantile(rel_np, q)) for q in _QUANTILE_LEVELS)


def _pick_compare_device(
    t_a: torch.Tensor,
    t_b: torch.Tensor,
    device: Union[str, torch.device] = "auto",
) -> torch.device:
    """Prefer CUDA when the pair fits; otherwise fall back to CPU to avoid OOM."""
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device != "auto":
        raise ValueError(f"device must be 'auto', 'cpu', 'cuda', or torch.device; got {device!r}")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if t_a.is_cuda:
        cuda_dev = t_a.device
    elif t_b.is_cuda:
        cuda_dev = t_b.device
    else:
        cuda_dev = torch.device("cuda", torch.cuda.current_device())
    need = t_a.numel() * _COMPARE_BYTES_PER_ELEM
    try:
        free, _total = torch.cuda.mem_get_info(cuda_dev.index)
    except RuntimeError:
        return torch.device("cpu")
    if need > int(free * _COMPARE_MEM_FRACTION):
        return torch.device("cpu")
    return cuda_dev


def _compute_row_metrics(
    t_a: torch.Tensor,
    t_b: torch.Tensor,
    param_name: str,
    *,
    device: Union[str, torch.device] = "auto",
) -> Dict[str, Any]:
    t_a = t_a.detach()
    t_b = t_b.detach()
    if t_a.shape != t_b.shape:
        sh = f"{tuple(t_a.shape)} vs {tuple(t_b.shape)}"
        return {
            "param_name": param_name,
            "ok_compare": False,
            "shape_mismatch": True,
            "is_empty": False,
            "shape": sh,
            "message": f"shape_mismatch: {sh}",
            "max_diff": float("nan"),
            "sum_abs_a": float("nan"),
            "sum_abs_b": float("nan"),
            "sum_rel_diff": float("nan"),
            "cosine_sim": float("nan"),
            "p95": float("nan"),
            "p90": float("nan"),
            "p80": float("nan"),
        }
    shape_str = str(tuple(t_a.shape))
    # Empty tensors: .max() without dim and cosine_similarity on 0-d vectors are ill-defined
    n = t_a.numel()
    if n == 0:
        return {
            "param_name": param_name,
            "ok_compare": True,
            "shape_mismatch": False,
            "is_empty": True,
            "shape": f"{shape_str} [numel=0]",
            "message": "numel=0 (both sides empty, same shape)",
            "max_diff": 0.0,
            "sum_abs_a": 0.0,
            "sum_abs_b": 0.0,
            "sum_rel_diff": 0.0,
            "cosine_sim": 1.0,
            "p95": 0.0,
            "p90": 0.0,
            "p80": 0.0,
        }

    # Per-tensor device: keep peak memory to one pair, not the whole state_dict.
    compare_device = _pick_compare_device(t_a, t_b, device)
    t_a = t_a.to(device=compare_device, dtype=torch.float32)
    t_b = t_b.to(device=compare_device, dtype=torch.float32)

    diff = (t_a - t_b).abs()
    max_diff = float(diff.max().item())
    sum_a = float(t_a.abs().sum().item())
    sum_b = float(t_b.abs().sum().item())
    denom = max(sum_a, sum_b) + 1e-20
    sum_rel_diff = abs(sum_a - sum_b) / denom

    # Identical tensors (incl. all-zero): skip quantile/cosine; cosine of zeros is ill-defined.
    if max_diff == 0.0:
        return {
            "param_name": param_name,
            "ok_compare": True,
            "shape_mismatch": False,
            "is_empty": False,
            "shape": shape_str,
            "message": "",
            "max_diff": 0.0,
            "sum_abs_a": sum_a,
            "sum_abs_b": sum_b,
            "sum_rel_diff": sum_rel_diff,
            "cosine_sim": 1.0,
            "p95": 0.0,
            "p90": 0.0,
            "p80": 0.0,
        }

    m = diff != 0
    if m.any():
        rel = diff[m] / (t_a[m].abs() + 1e-20)
        p95, p90, p80 = _rel_error_quantiles(rel)
    else:
        p95 = p90 = p80 = 0.0
    cos = float(
        F.cosine_similarity(
            t_a.reshape(1, -1), t_b.reshape(1, -1), dim=1, eps=1e-20
        ).item()
    )
    return {
        "param_name": param_name,
        "ok_compare": True,
        "shape_mismatch": False,
        "is_empty": False,
        "shape": shape_str,
        "message": "",
        "max_diff": max_diff,
        "sum_abs_a": sum_a,
        "sum_abs_b": sum_b,
        "sum_rel_diff": sum_rel_diff,
        "cosine_sim": cos,
        "p95": p95,
        "p90": p90,
        "p80": p80,
    }


def tensor_precision_report_rows(
    a: Union[torch.Tensor, Mapping[str, torch.Tensor]],
    b: Union[torch.Tensor, Mapping[str, torch.Tensor]],
    *,
    name: str = "tensor",
    thresholds: Optional[PrecisionReportThresholds] = None,
    device: Union[str, torch.device] = "auto",
) -> List[Dict[str, Any]]:
    """
    Build per-parameter metric dicts (same comparison logic as :func:`tensor_precision_report`).
    Use this when you need structured results; use :func:`tensor_precision_report` for the
    rendered table string.

    Args:
        device: Where to run comparisons. ``"auto"`` uses CUDA when the pair fits in free
            memory, otherwise CPU. ``"cpu"`` / ``"cuda"`` force a device.
    """
    th = thresholds if thresholds is not None else PrecisionReportThresholds()
    rows: List[Dict[str, Any]] = []

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        m = _compute_row_metrics(a, b, name, device=device)
        m["aligned"] = th.is_aligned(m)
        rows.append(m)
    elif isinstance(a, Mapping) and isinstance(b, Mapping):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        for k in only_a:
            ta = a[k]
            sh = str(tuple(ta.shape)) if isinstance(ta, torch.Tensor) else "?"
            rows.append(
                {
                    "param_name": k,
                    "ok_compare": False,
                    "shape_mismatch": False,
                    "is_empty": bool(isinstance(ta, torch.Tensor) and ta.numel() == 0),
                    "shape": sh,
                    "message": "missing in b",
                    "max_diff": float("nan"),
                    "sum_abs_a": float("nan"),
                    "sum_abs_b": float("nan"),
                    "sum_rel_diff": float("nan"),
                    "cosine_sim": float("nan"),
                    "p95": float("nan"),
                    "p90": float("nan"),
                    "p80": float("nan"),
                    "aligned": False,
                }
            )
        for k in only_b:
            tb = b[k]
            sh = str(tuple(tb.shape)) if isinstance(tb, torch.Tensor) else "?"
            rows.append(
                {
                    "param_name": k,
                    "ok_compare": False,
                    "shape_mismatch": False,
                    "is_empty": bool(isinstance(tb, torch.Tensor) and tb.numel() == 0),
                    "shape": sh,
                    "message": "missing in a",
                    "max_diff": float("nan"),
                    "sum_abs_a": float("nan"),
                    "sum_abs_b": float("nan"),
                    "sum_rel_diff": float("nan"),
                    "cosine_sim": float("nan"),
                    "p95": float("nan"),
                    "p90": float("nan"),
                    "p80": float("nan"),
                    "aligned": False,
                }
            )
        for k in sorted(keys_a & keys_b):
            m = _compute_row_metrics(a[k], b[k], k, device=device)
            m["aligned"] = th.is_aligned(m)
            rows.append(m)
    else:
        raise TypeError(
            "a and b must be both torch.Tensor or both dict-like; "
            f"got {type(a)} and {type(b)}"
        )
    return rows


def tensor_precision_report(
    a: Union[torch.Tensor, Mapping[str, torch.Tensor]],
    b: Union[torch.Tensor, Mapping[str, torch.Tensor]],
    *,
    name: str = "tensor",
    thresholds: Optional[PrecisionReportThresholds] = None,
    print_out: bool = True,
    sort_by: Optional[str] = 'max_diff',
    device: Union[str, torch.device] = "auto",
) -> str:
    """
    Compare two tensors or two state_dicts and return the same text as the table report
    (optionally also printed to stdout).

    For each key (or the single pair), reports max_diff, L1 sums, cosine_sim, p95/p90/p80
    relative error (on mismatched elements), and whether thresholds pass. If both sides
    are element-wise equal (e.g. all-zero frozen weights/grads), all scalars can be 0
    and cosine is reported as 1 (not 0) because the cosine of two all-zero vectors is
    ill-defined in ``F.cosine_similarity``.     For rows that are not fully aligned, ``param_name`` and ``aligned`` (last column) are
    styled red; ``shape`` is red only when the two tensors have different shapes for the
    same key; and each metric cell that violates its threshold (e.g. ``p90``) is also red.

    Args:
        a: First tensor or mapping of name -> tensor.
        b: Second tensor or mapping. For mappings, only keys present in *both* are
           compared. Keys only in one side are reported as missing (not aligned).
        name: Parameter name when ``a`` and ``b`` are plain tensors.
        thresholds: Criterion for green vs red. Defaults to :class:`PrecisionReportThresholds`.
        print_out: If True, print the report to stdout.
        device: Where to run comparisons. ``"auto"`` uses CUDA when the pair fits in free
            memory, otherwise CPU. ``"cpu"`` / ``"cuda"`` force a device.

    Returns:
        The full report string (table + summary), with selective red on failing cells as
        above. For structured per-parameter dicts, use :func:`tensor_precision_report_rows`.
    """
    th = thresholds if thresholds is not None else PrecisionReportThresholds()
    rows = tensor_precision_report_rows(
        a, b, name=name, thresholds=th, device=device
    )
    report = _format_precision_report_table(rows, th, sort_by=sort_by)
    if print_out:
        print(report, end="")
    return report


# Single source of column order: row cell lists in ``_format_precision_report_table`` must
# match this list left-to-right. New thresholded metrics: add a name here, append the cell
# in both ok_compare branches, and add the same key in ``failed_metric_table_columns``.
PRECISION_REPORT_TABLE_COLS: Tuple[str, ...] = (
    "param_name",
    "shape",
    "max_diff",
    "sum_abs_a",
    "sum_abs_b",
    "sum_rel_diff",
    "cosine_sim",
    "p95",
    "p90",
    "p80",
    "aligned",
)

def _format_precision_report_table(
    rows: List[Dict[str, Any]], th: PrecisionReportThresholds, sort_by: Optional[str] = None,
) -> str:
    """
    Not styling by magic indices: ``param_name`` / ``shape`` / ``aligned`` and threshold
    columns are resolved via :data:`PRECISION_REPORT_TABLE_COLS` so re-ordering or inserting
    columns only requires keeping that tuple and the cell builders in sync.
    """
    if not rows:
        return "tensor_precision_report: (empty)\n"

    if sort_by:
        def get_value(x: Dict[str, Any]) -> Any:
            return x.get(sort_by, float("nan"))
        rows.sort(key=get_value, reverse=True)

    col_names = list(PRECISION_REPORT_TABLE_COLS)

    str_rows: List[List[str]] = []
    aligns: List[bool] = []
    for r in rows:
        shape_cell = str(r.get("shape", "-"))[:56]
        if r.get("ok_compare", True):
            cells = [
                r["param_name"][:64],
                shape_cell,
                f"{r['max_diff']:.4e}",
                f"{r['sum_abs_a']:.3e}",
                f"{r['sum_abs_b']:.3e}",
                f"{r['sum_rel_diff']:.4e}",
                f"{r['cosine_sim']:.4e}",
                f"{r['p95']:.3e}",
                f"{r['p90']:.3e}",
                f"{r['p80']:.3e}",
                str(r.get("aligned", False)),
            ]
        else:
            msg = r.get("message", "")
            cells = [
                str(r["param_name"])[:64],
                shape_cell,
                msg[:48],
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "False",
            ]
        str_rows.append(cells)
        aligns.append(r.get("aligned", False) and (r.get("ok_compare", True)))

    for row in str_rows:
        if len(row) != len(col_names):
            raise ValueError(
                f"row has {len(row)} cells but PRECISION_REPORT_TABLE_COLS has {len(col_names)} "
                f"— update the cell lists in lockstep with {PRECISION_REPORT_TABLE_COLS!r}."
            )

    widths = [len(cn) for cn in col_names]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    name_to_i = {n: i for i, n in enumerate(col_names)}
    # Structural columns for highlight rules (by name, not 0/1/last)
    try:
        i_param = name_to_i["param_name"]
        i_shape = name_to_i["shape"]
        i_aligned = name_to_i["aligned"]
    except KeyError as e:
        raise KeyError(
            f"PRECISION_REPORT_TABLE_COLS must include param_name, shape, aligned: {e}"
        ) from e
    widths[i_param] = min(max(widths[i_param], 32), 72)

    def fmt_line(
        cells: List[str],
        not_ok: bool,
        shape_mismatch: bool,
        row_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        if len(cells) != len(col_names):
            raise ValueError("fmt_line cells length does not match col_names")
        parts: List[str] = []
        fail_names = (
            th.failed_metric_table_columns(row_dict)
            if (not_ok and row_dict and row_dict.get("ok_compare", True))
            else set()
        )
        fail_idx = {name_to_i[n] for n in fail_names if n in name_to_i}
        for i, cell in enumerate(cells):
            pad = cell.ljust(widths[i])
            red = False
            if not_ok:
                if i == i_param or i == i_aligned:
                    red = True
                elif i == i_shape and shape_mismatch:
                    red = True
                elif i in fail_idx:
                    red = True
            if red:
                parts.append(console.colorize("red", pad))
            else:
                parts.append(pad)
        return "  ".join(parts)

    header = fmt_line(
        col_names, not_ok=False, shape_mismatch=False, row_dict=None
    )
    n_ok = sum(1 for r in rows if r.get("aligned", False) and r.get("ok_compare", True))
    out_lines: List[str] = [
        header,
        "-" * min(len(header), 200),
    ]
    for row_dict, row_cells, ok in zip(rows, str_rows, aligns):
        out_lines.append(
            fmt_line(
                row_cells,
                not_ok=not ok,
                shape_mismatch=bool(row_dict.get("shape_mismatch", False)),
                row_dict=row_dict,
            )
        )
    out_lines.append("-" * min(len(header), 200))
    out_lines.append(
        f"summary: {n_ok}/{len(rows)} aligned  (thresholds: max_diff<={th.max_diff}, "
        f"p95/p90/p80<={th.p95}/{th.p90}/{th.p80}, cosine>={th.cosine_min}"
        + (
            f", sum_rel_diff<={th.sum_rel_diff_max}"
            if th.sum_rel_diff_max is not None
            else ", sum_rel_diff: (not used)"
        )
        + ")\n"
    )
    return "\n".join(out_lines)


