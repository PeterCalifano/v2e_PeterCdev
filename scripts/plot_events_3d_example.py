#!/usr/bin/env python3
"""Generate dummy motion frames, run v2e emulator, and plot events in 3D (t, x, y).

Examples
--------
python scripts/plot_events_3d_example.py --scenario moving_edge
python scripts/plot_events_3d_example.py --scenario moving_blob --save output/events_3d_blob.png --no_show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2ecore.emulator import EventEmulator


def _configure_matplotlib(no_show: bool, backend: str) -> str:
    import matplotlib

    if no_show:
        matplotlib.use("Agg", force=True)
        return "Agg"

    if backend != "auto":
        matplotlib.use(backend, force=True)
        return backend

    # Prefer a non-Qt backend to avoid Qt/xcb plugin conflicts from cv2 builds.
    for candidate in ("TkAgg", "Agg"):
        try:
            matplotlib.use(candidate, force=True)
            return candidate
        except Exception:
            continue

    # Final fallback.
    matplotlib.use("Agg", force=True)
    return "Agg"


def _build_moving_edge_frames(
    num_frames: int, height: int, width: int, shift_px: int
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    x = np.arange(width, dtype=np.int32)[None, :].repeat(height, axis=0)
    base = np.where(x < (width // 2), 255, 0).astype(np.uint8)
    for i in range(num_frames):
        frames.append(np.roll(base, shift=i * shift_px, axis=1))
    return frames


def _build_moving_blob_frames(
    num_frames: int, height: int, width: int, shift_px: int
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    blob_h = max(6, height // 4)
    blob_w = max(8, width // 5)
    y0 = max(2, (height - blob_h) // 2)
    x0 = max(2, width // 8)

    for i in range(num_frames):
        frame = np.zeros((height, width), dtype=np.uint8)
        xi = (x0 + i * shift_px) % max(1, width - blob_w)
        frame[y0:y0 + blob_h, xi:xi + blob_w] = 220
        frames.append(frame)
    return frames


def _build_frames(
    scenario: str, num_frames: int, height: int, width: int, shift_px: int
) -> list[np.ndarray]:
    if scenario == "moving_edge":
        return _build_moving_edge_frames(num_frames, height, width, shift_px)
    if scenario == "moving_blob":
        return _build_moving_blob_frames(num_frames, height, width, shift_px)
    raise ValueError(f"Unknown scenario: {scenario}")


def _scene_montage(frames: list[np.ndarray]) -> np.ndarray:
    """Build a simple first/middle/last grayscale montage for 2D scene context."""
    if len(frames) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    first = frames[0]
    mid = frames[len(frames) // 2]
    last = frames[-1]
    h = first.shape[0]
    sep = np.full((h, 2), 128, dtype=np.uint8)
    return np.concatenate([first, sep, mid, sep, last], axis=1)


def _generate_events(
    frames: list[np.ndarray],
    fps: float,
    width: int,
    height: int,
    device: str,
    seed: int,
) -> np.ndarray:
    
    emu = EventEmulator(
        pos_thres=0.15,
        neg_thres=0.15,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=seed,
        output_width=width,
        output_height=height,
        device=device,
    )

    dt = 1.0 / fps
    t = 0.0
    chunks: list[np.ndarray] = []

    with torch.no_grad():
        for frame in frames:
            ev = emu.generate_events(frame, t)
            if ev is not None and ev.shape[0] > 0:
                chunks.append(ev)
            t += dt

    emu.cleanup()

    if len(chunks) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dummy-frame v2e example with 3D event plot (t, x, y).")
    parser.add_argument(
        "--scenario",
        choices=["moving_edge", "moving_blob"],
        default="moving_blob",
    )
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--shift_px", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--mpl_backend",
        default="auto",
        help="Matplotlib backend. Use 'auto' (default), or e.g. TkAgg/Agg.",
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no_show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 1:
        raise ValueError("--frames must be > 1")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available")

    selected_backend = _configure_matplotlib(
        no_show=args.no_show, backend=args.mpl_backend
    )
    import matplotlib.pyplot as plt

    frames = _build_frames(
        scenario=args.scenario,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        shift_px=args.shift_px,
    )
    events = _generate_events(
        frames=frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
        device=args.device,
        seed=args.seed,
    )

    print(
        f"[3D PLOT] scenario={args.scenario} size={args.width}x{args.height} "
        f"frames={args.frames} events={events.shape[0]} backend={selected_backend}"
    )

    fig = plt.figure(figsize=(14, 6))
    ax_scene = fig.add_subplot(1, 2, 1)
    ax = fig.add_subplot(1, 2, 2, projection="3d")

    montage = _scene_montage(frames)
    ax_scene.imshow(montage, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_scene.set_title("2D scene snapshots: first | mid | last")
    ax_scene.set_xlabel("x (px)")
    ax_scene.set_ylabel("y (px)")

    if events.shape[0] > 0:
        on_mask = events[:, 3] > 0
        off_mask = ~on_mask
        if np.any(on_mask):
            ax.scatter(
                events[on_mask, 0],
                events[on_mask, 1],
                events[on_mask, 2],
                s=3,
                c="tab:red",
                alpha=0.7,
                label="ON (+1)",
            )
        if np.any(off_mask):
            ax.scatter(
                events[off_mask, 0],
                events[off_mask, 1],
                events[off_mask, 2],
                s=3,
                c="tab:blue",
                alpha=0.7,
                label="OFF (-1)",
            )
    else:
        ax.text2D(0.05, 0.95, "No events generated", transform=ax.transAxes)

    ax.set_xlabel("t (s)")
    ax.set_ylabel("x (px)")
    ax.set_zlabel("y (px)")
    ax.set_title(f"v2e events in (t, x, y) - {args.scenario}")
    ax.legend(loc="upper right")
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"[3D PLOT] saved figure: {args.save}")

    if not args.no_show and selected_backend != "Agg":
        plt.show()
    elif not args.no_show and selected_backend == "Agg":
        print("[3D PLOT] Agg backend selected; no interactive window shown.")
    plt.close(fig)


if __name__ == "__main__":
    main()
