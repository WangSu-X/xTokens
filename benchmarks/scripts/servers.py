"""Start the xTokens server for benchmark runs."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .support import INSTRUMENTS, ROOT, write_json


def command(runtime: dict, python: str, port: int) -> list[str]:
    return [
        python,
        "-m",
        "x_tokens",
        "--model",
        runtime["model"],
        "--served-model-name",
        runtime["served_model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        str(runtime["max_model_len"]),
        "--max-num-seqs",
        str(runtime["max_num_seqs"]),
        "--kv-cache-memory-bytes",
        str(runtime["kv_cache_memory_bytes"]),
        "--local-files-only",
        "--no-access-log",
        "--kv-cache-block-size",
        "128",
        "--disable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-prefill-batch-tokens",
        str(runtime["xtokens_prefill_batch_tokens"]),
        "--prefill-chunk-alignment",
        str(runtime["xtokens_prefill_chunk_alignment"]),
    ]


def gpu_snapshot(runtime: dict) -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ""
    return subprocess.check_output(
        [
            executable,
            "--id",
            ",".join(map(str, runtime["gpus"])),
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used",
            "--format=csv",
        ],
        text=True,
    )


@contextlib.contextmanager
def serving_engine(
    runtime: dict,
    output: Path,
    *,
    python: str,
    port: int,
    model_only: bool = False,
):
    output.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}"
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"benchmark port {port} already has a listener")
    argv = command(runtime, python, port)
    env = os.environ.copy()
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=",".join(map(str, runtime["gpus"])),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="2",
    )
    paths = [str(ROOT.parent)]
    if model_only:
        paths.insert(0, str(INSTRUMENTS))
        env["XTOKENS_BENCH_CONTROL"] = str(output / "model-control.json")
        (output / "model-control.json").unlink(missing_ok=True)
    env["PYTHONPATH"] = os.pathsep.join(
        paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    write_json(
        output / "launch.json",
        {
            "argv": argv,
            "runtime": runtime,
            "pythonpath": paths,
            "model_only": model_only,
        },
    )
    with (output / "server.log").open("w") as stream:
        process = subprocess.Popen(
            argv,
            cwd=ROOT.parent,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 300
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(url + "/ready", timeout=2):
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise RuntimeError(
                    f"xTokens failed to become ready; see {output / 'server.log'}"
                )
            (output / "gpus.csv").write_text(gpu_snapshot(runtime))
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
