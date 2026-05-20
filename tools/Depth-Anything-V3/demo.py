"""Depth Anything 3 video demo.

Runs DA3 over every frame of an input video and saves the colorized depth
maps as PNGs. Used as a sanity check for the DA3 install and to seed the
REVPT depth_estimation tool.

Usage:
    # From the repo root, all 72 frames of simulation_0311.mp4:
    python tools/Depth-Anything-V3/demo.py simulation_0311.mp4

    # Subsample, custom output, metric variant:
    python tools/Depth-Anything-V3/demo.py simulation_0311.mp4 \
        --output depth_outputs/ \
        --every 4 \
        --model-id depth-anything/DA3METRIC-LARGE
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib
import numpy as np
import torch
from PIL import Image


def load_model(model_id: str = "depth-anything/DA3MONO-LARGE", device: str | None = None):
    """Load Depth Anything 3 once. Returns (model, device_str)."""
    from depth_anything_3.api import DepthAnything3

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[demo] loading {model_id} on {device} ...", flush=True)
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=torch.device(device))
    return model, device


def depth_estimation(
    image: Image.Image,
    model,
    cmap_name: str = "Spectral_r",
) -> Image.Image:
    """Run DA3 depth estimation on a single PIL image and return a colorized
    PIL image of the same H, W. This is the function that should later be
    wired up as the REVPT `depth_estimation` tool.
    """
    # DA3's inference API takes a list of file paths, so we persist the
    # frame to a temp file. Could be replaced by an in-memory variant if
    # upstream adds one.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        with torch.no_grad():
            prediction = model.inference([tmp_path])
        depth = prediction.depth[0]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    span = float(depth.max() - depth.min())
    norm = (depth - depth.min()) / (span + 1e-9)
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


def iter_video_frames(video_path: str, stride: int = 1) -> Iterator[Tuple[int, Image.Image]]:
    """Yield (frame_index, PIL.Image RGB) for every `stride` frames of the video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[demo] {video_path}: {total} frames @ {fps:.2f} fps; stride={stride}", flush=True)

    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
            idx += 1
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to input video (e.g. simulation_0311.mp4)")
    parser.add_argument("--output", default="depth_outputs", help="Output directory (default: depth_outputs)")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame (default: 1 = every frame)")
    parser.add_argument(
        "--model-id",
        default="depth-anything/DA3MONO-LARGE",
        help="HuggingFace model id (default: depth-anything/DA3MONO-LARGE)",
    )
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"], help="Force device (default: auto)")
    parser.add_argument("--cmap", default="Spectral_r", help="matplotlib colormap (default: Spectral_r)")
    args = parser.parse_args(argv)

    if not Path(args.video).exists():
        print(f"[demo] error: {args.video} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_model(args.model_id, args.device)

    n_saved = 0
    for frame_idx, rgb_image in iter_video_frames(args.video, stride=args.every):
        depth_image = depth_estimation(rgb_image, model, cmap_name=args.cmap)
        out_path = out_dir / f"frame_{frame_idx:04d}_depth.png"
        depth_image.save(out_path)
        n_saved += 1
        if n_saved % 10 == 0 or n_saved == 1:
            print(f"[demo] saved {out_path}", flush=True)

    print(f"[demo] done. {n_saved} depth PNGs written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
