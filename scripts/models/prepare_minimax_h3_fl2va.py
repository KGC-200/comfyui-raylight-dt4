#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys

BASE_MODELS_DIR = "/tmp/comfy_models"
COMFY_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

WEIGHTS_MANIFEST = [
    {
        "repo": "Comfy-Org/MiniMax-H3",
        "file": "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "target_subfolder": "vae",
        "target_name": "minimax_h3_video_vae_int8_convrot.safetensors",
    },
    {
        "repo": "Comfy-Org/MiniMax-H3",
        "file": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "target_subfolder": "vae",
        "target_name": "minimax_h3_audio_vae_fp32.safetensors",
    },
    {
        "repo": "lightx2v/Minimax-h3-Turbo",
        "file": "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors",
        "target_subfolder": "loras",
        "target_name": "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors",
    },
    {
        "repo": "lightx2v/Minimax-h3-Turbo",
        "file": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "target_subfolder": "loras",
        "target_name": "minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
    },
    {
        "repo": "Comfy-Org/MiniMax-H3",
        "file": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "target_subfolder": "diffusion_models",
        "target_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    },
    {
        "repo": "Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI",
        "file": "qwen3vl_32b_minimax_h3-int4_convrot.safetensors",
        "target_subfolder": "text_encoders",
        "target_name": "qwen3vl_32b_minimax_h3-int4_convrot.safetensors",
    },
]


def setup_paths():
    subfolders = ["diffusion_models", "vae", "loras", "text_encoders"]
    comfy_models = os.path.join(COMFY_APP_DIR, "models")
    os.makedirs(comfy_models, exist_ok=True)

    for sub in subfolders:
        tmp_target = os.path.join(BASE_MODELS_DIR, sub)
        os.makedirs(tmp_target, exist_ok=True)
        link_target = os.path.join(comfy_models, sub)
        if os.path.islink(link_target) or os.path.exists(link_target):
            try:
                if os.path.islink(link_target):
                    os.unlink(link_target)
                else:
                    shutil.rmtree(link_target)
            except Exception:
                pass
        try:
            os.symlink(tmp_target, link_target)
        except Exception:
            pass


def clean_cache():
    cache_dirs = [
        os.path.join(BASE_MODELS_DIR, ".cache"),
        "/root/.cache/huggingface",
        "/tmp/.cache",
    ]
    for c in cache_dirs:
        if os.path.exists(c):
            try:
                shutil.rmtree(c)
            except Exception:
                pass


def fetch_file(repo: str, remote_file: str, target_subfolder: str, target_name: str):
    target_dir = os.path.join(BASE_MODELS_DIR, target_subfolder)
    dest_path = os.path.join(target_dir, target_name)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return

    hf_bin = "/tmp/.venv/bin/hf"
    env = os.environ.copy()
    env["HF_XET_HIGH_PERFORMANCE"] = "1"

    cmd = [
        hf_bin if os.path.exists(hf_bin) else "hf",
        "download",
        repo,
        remote_file,
        "--local-dir",
        target_dir,
    ]

    try:
        subprocess.run(cmd, env=env, check=True)
    except Exception:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo, filename=remote_file, local_dir=target_dir)

    downloaded_nested = os.path.join(target_dir, remote_file)
    if os.path.exists(downloaded_nested) and downloaded_nested != dest_path:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(downloaded_nested, dest_path)

    clean_cache()


def main():
    setup_paths()

    total = len(WEIGHTS_MANIFEST)
    for i, item in enumerate(WEIGHTS_MANIFEST, 1):
        name = item["target_name"]
        print(f"[{i}/{total}] {name}...")
        try:
            fetch_file(
                repo=item["repo"],
                remote_file=item["file"],
                target_subfolder=item["target_subfolder"],
                target_name=name,
            )
            print("Listo.")
        except Exception as e:
            print(f"Error: {e}")

    clean_cache()
    print("Completado.")


if __name__ == "__main__":
    main()
