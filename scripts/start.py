#!/usr/bin/env python3

import os
import re
import subprocess
import sys
import time
import urllib.request


def cleanup():
    targets = ["main.py", "ray::", "gcs_server", "raylet", "cloudflared"]
    for t in targets:
        os.system(f"pkill -9 -f '{t}' 2>/dev/null || true")
    time.sleep(2)


def get_env():
    env = os.environ.copy()
    python_bin = "/tmp/.venv/bin/python"
    nccl_lib = "/tmp/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"

    env["VIRTUAL_ENV"] = "/tmp/.venv"
    env["PATH"] = f"/tmp/.venv/bin:{env.get('PATH', '')}"
    env["UV_PYTHON"] = python_bin
    env["LD_LIBRARY_PATH"] = f"{nccl_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["RAYLIGHT_KEEP_CUDA_MALLOC_ASYNC"] = "0"
    env["NCCL_CUMEM_ENABLE"] = "0"
    env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_SHM_DISABLE"] = "1"

    return env, python_bin


def run_core(env, python_bin, app_dir):
    log_file = open("/tmp/core.log", "w")
    cmd = [
        python_bin,
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--disable-pinned-memory",
        "--reserve-vram",
        "0.5",
        "--cache-none",
        "--enable-manager",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=app_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    for _ in range(45):
        time.sleep(2)
        try:
            if urllib.request.urlopen("http://127.0.0.1:8188", timeout=2).getcode() == 200:
                return proc
        except Exception:
            pass
    return proc


def run_proxy():
    bin_path = "/tmp/cloudflared-linux-amd64"
    if not os.path.exists(bin_path):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        os.system(f"wget -q {url} -O {bin_path}")
        os.system(f"chmod +x {bin_path}")

    log_path = "/tmp/proxy.log"
    log_file = open(log_path, "w")

    cmd = [
        bin_path,
        "tunnel",
        "--url",
        "http://127.0.0.1:8188",
        "--protocol",
        "http2",
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    url_found = None
    for _ in range(30):
        time.sleep(1)
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
                if matches:
                    url_found = matches[0]
                    break

    return proc, url_found


def main():
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cleanup()

    env, python_bin = get_env()
    core_proc = run_core(env, python_bin, app_dir)
    proxy_proc, url = run_proxy()

    if url:
        print(f"Listo: {url}")
    else:
        print("Error")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        core_proc.terminate()
        proxy_proc.terminate()


if __name__ == "__main__":
    main()
