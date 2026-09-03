
import os
import re
import subprocess

# CPU Affinity column only contains ranges like "0-55,112-167", not NUMA id "0"/"1".
_CPU_AFFINITY_RE = re.compile(r"\d+-\d+(?:,\d+-\d+)*")
_GPU_ROW_RE = re.compile(r"^GPU(\d+)\s+\S")


def _parse_cpu_list(cpu_list: str) -> list[int]:
    cpus: list[int] = []
    for part in cpu_list.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            cpus.extend(range(start, end + 1))
        elif part.isdigit():
            cpus.append(int(part))
    return sorted(set(cpus))


def _parse_gpu_cpu_affinity(topo: str) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for raw_line in topo.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).strip()
        m = _GPU_ROW_RE.match(line)
        if not m or line.split()[1].startswith("GPU"):  # skip header row
            continue
        cpu_m = _CPU_AFFINITY_RE.search(line)
        if cpu_m:
            result[int(m.group(1))] = _parse_cpu_list(cpu_m.group())
    return result


def bind_cpu_for_local_rank(verbose: bool = False) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    topo = subprocess.check_output(
        ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL,
    )
    gpu_affinity = _parse_gpu_cpu_affinity(topo)

    if local_rank not in gpu_affinity:
        raise RuntimeError(
            f"Cannot find CPU affinity for GPU{local_rank}. Parsed: {gpu_affinity}"
        )

    my_pool = gpu_affinity[local_rank]
    same_pool = sorted(
        gpu for gpu in range(local_world_size) if gpu_affinity.get(gpu) == my_pool
    )
    cpus = set(my_pool[same_pool.index(local_rank) :: len(same_pool)])

    os.sched_setaffinity(0, cpus)
    if verbose:
        print(f"[LOCAL_RANK={local_rank}] bind CPUs: {sorted(cpus)}", flush=True)


if 'LOCAL_RANK' in os.environ:
    try:
        bind_cpu_for_local_rank()
    except Exception as e:
        ...
