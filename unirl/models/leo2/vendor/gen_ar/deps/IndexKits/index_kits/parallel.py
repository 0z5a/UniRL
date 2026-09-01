# ==============================================================================
# Implement a common interface for parallel processing of tasks using multiple
# nodes.
# ==============================================================================

import os
import sys
import subprocess
from pathlib import Path


def run_parallel(hostfile, argv, master_addr=None, master_port=13579):
    hostfile = Path(hostfile)
    if not hostfile.exists():
        raise FileNotFoundError(f"Hostfile {hostfile} does not exist.")

    # Make sure deepspeed is installed
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        raise ImportError("DeepSpeed is not installed. Please install it to use multi-node processing.")

    if master_addr is None:
        master_addr = os.getenv("CHIEF_IP")
    if master_addr is None:
        master_addr = os.getenv("LOCAL_IP")
    if master_addr is None:
        raise ValueError("Either --master-addr or CHIEF_IP or LOCAL_IP environment variable must be set.")

    # Find idk path
    idk_entry = Path(__file__).parents[1] / "bin/idk"

    cmd = [
        "deepspeed",
        "--hostfile", str(hostfile),
        "--master_addr", f"{master_addr}", "--master_port", f"{master_port}",
        str(idk_entry),
    ] + argv

    print("Running command:", " ".join(cmd))
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Parallel processing failed with return code {process.returncode}.")

    # Exit
    sys.exit(process.returncode)
