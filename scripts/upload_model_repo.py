"""Create or update the Hugging Face model repo (weights, ONNX, model card, examples).

HF_TOKEN is read from the environment or .env and is never printed. Without a token the script says so and exits 0.
The model repository is published under AGPL-3.0:
https://huggingface.co/Aarun-1/defect-detector-neu-det

Usage:
  python scripts/upload_model_repo.py --stage-only     # build upload folder under build/hf, no network
  python scripts/upload_model_repo.py                  # upload model repo (needs HF_TOKEN)
  python scripts/upload_model_repo.py --onnx-only      # only add models/best.onnx to the model repo
  python scripts/upload_model_repo.py --onnx-only --examples   # also add 6 val example images
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit_app.core import pick_examples  # noqa: E402

MODEL_COPY = [
    ("models/best.pt", "best.pt"),
    ("MODEL_CARD.md", "README.md"),
    ("reports/phase3/frozen.json", "policy/frozen.json"),
    ("configs/cost.yaml", "policy/cost.yaml"),
]


def get_token(env_file: Path = ROOT / ".env") -> str:
    """HF_TOKEN from the environment, else from .env; empty string if neither has one."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok and env_file.is_file():
        from dotenv import dotenv_values

        tok = (dotenv_values(env_file).get("HF_TOKEN") or "").strip()
    return tok


def scrub(text: str, token: str) -> str:
    """Remove the token from any message before it is printed."""
    return text.replace(token, "<token>") if token else text


def _copy(pairs, dest: Path) -> None:
    for src, dst in pairs:
        s = ROOT / src
        if not s.is_file():
            raise FileNotFoundError(f"{s} is needed for the upload but missing")
        (dest / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dest / dst)


def stage(out: Path, model_repo: str) -> Path:
    """Build the model-repo upload folder under `out`; returns model_dir."""
    if out.exists():
        shutil.rmtree(out)
    model_dir = out / "model"
    _copy(MODEL_COPY, model_dir)
    onnx = ROOT / "models/best.onnx"
    if onnx.is_file():
        shutil.copy2(onnx, model_dir / "best.onnx")
    val_images = ROOT / "data/yolo/images/val"
    if val_images.is_dir():
        ex = model_dir / "examples"
        ex.mkdir(parents=True, exist_ok=True)
        for p in pick_examples(str(val_images), 6):
            shutil.copy2(p, ex / p.name)
    return model_dir


def upload_onnx(api, model_repo: str, examples: bool = False) -> int:
    """Add best.onnx (and optionally example val images) to an existing model repo without touching other files."""
    onnx = ROOT / "models/best.onnx"
    if not onnx.is_file():
        print("models/best.onnx is missing; run: python -m src.onnx_detector export")
        return 1
    api.upload_file(
        path_or_fileobj=onnx,
        path_in_repo="best.onnx",
        repo_id=model_repo,
        repo_type="model",
        commit_message="Add ONNX export (opset 17, 640x640, CPU)",
    )
    if examples:
        val_images = ROOT / "data/yolo/images/val"
        if val_images.is_dir():
            for p in pick_examples(str(val_images), 6):
                api.upload_file(
                    path_or_fileobj=p,
                    path_in_repo=f"examples/{p.name}",
                    repo_id=model_repo,
                    repo_type="model",
                    commit_message="Add val example image",
                )
    print(f"model: https://huggingface.co/{model_repo}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", help="HF user or org (default: the token's user)")
    ap.add_argument("--model-name", default="defect-detector-neu-det")
    ap.add_argument("--out", default="build/hf")
    ap.add_argument("--onnx-only", action="store_true", help="upload models/best.onnx to the model repo and stop")
    ap.add_argument("--examples", action="store_true", help="with --onnx-only: also upload 6 val example images")
    ap.add_argument("--stage-only", action="store_true", help="build the folders and stop; no token, no network")
    a = ap.parse_args()

    token = get_token()
    if a.stage_only:
        model_dir = stage(Path(a.out), f"<namespace>/{a.model_name}")
        print(model_dir, sorted(str(p.relative_to(model_dir)) for p in model_dir.rglob("*") if p.is_file()))
        return 0
    if not token:
        print("HF_TOKEN is not set (environment or .env); nothing deployed.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        ns = a.namespace or api.whoami()["name"]
        model_repo = f"{ns}/{a.model_name}"
        if a.onnx_only:
            return upload_onnx(api, model_repo, a.examples)
        model_dir = stage(Path(a.out), model_repo)
        api.create_repo(model_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=model_repo,
            repo_type="model",
            folder_path=model_dir,
            commit_message="Upload weights, ONNX, model card, examples and frozen policy",
        )
    except Exception as exc:  # network, permission or validation errors; never echo the token
        print(f"deploy failed: {type(exc).__name__}: {scrub(str(exc), token)}")
        return 1
    print(f"model: https://huggingface.co/{model_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
