import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from simulator.events import run_kmc

def parse_optional_float(text):
    value = str(text).strip().lower()
    if value in {"none", "null"}:
        return None
    return float(text)

def build_parser():
    parser = argparse.ArgumentParser(description="Generate a movie for one kMC simulation")
    parser.add_argument("--F", type=float, default=0.40, help="Flux")
    parser.add_argument("--E_d", type=float, default=0.50, help="Diffusion barrier")
    parser.add_argument("--E_des", type=float, default=2.20, help="Desorption barrier")
    parser.add_argument("--T", type=float, default=320.0, help="Temperature")   
    parser.add_argument("--n", type=int, default=100, help="Lattice side length")
    parser.add_argument("--max-steps", type=int, default=1_500_000, help="Maximum kMC steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--time-factor", type=float, default=5.0, help="Stop time factor")
    parser.add_argument("--target-coverage", type=parse_optional_float, default=None, help="Optional early stop target")
    parser.add_argument("--max-diff-to-ads-ratio", type=parse_optional_float, default=120.0, help="Optional diffusion cap")        
    parser.add_argument("--immobile-if-neighbors-ge", type=int, default=2, help="Stability rule: lock atoms with >= N occupied neighbors")
    parser.add_argument("--snapshot-every", type=int, default=1, help="Capture one frame every N steps")     
    parser.add_argument("--fps", type=int, default=20, help="Animation fps")    
    parser.add_argument("--dpi", type=int, default=120, help="Output dpi")      
    parser.add_argument("--bitrate", type=int, default=1800, help="Bitrate for mp4 output")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--out", type=str, default="movies/single_sim.gif", help="Output path")
    return parser

def main():
    args = build_parser().parse_args()
    params = {"F": args.F, "E_d": args.E_d, "E_des": args.E_des, "T": args.T}

    frames = []
    frame_steps = []
    frame_times = []

    def capture(step, time, grid, grain_ids, n):
        frames.append(grid.reshape(n, n).copy())
        frame_steps.append(int(step))
        frame_times.append(float(time))

    cov, gbd, sim_time = run_kmc(
        params=params, max_steps=args.max_steps, n=args.n, seed=args.seed,
        time_factor=args.time_factor, target_coverage=args.target_coverage,
        max_diff_to_ads_ratio=args.max_diff_to_ads_ratio,
        immobile_if_neighbors_ge=args.immobile_if_neighbors_ge if hasattr(args, "immobile_if_neighbors_ge") else 2,
        snapshot_every_steps=args.snapshot_every, snapshot_callback=capture,
    )

    if not frames:
        raise RuntimeError("No frames were captured.")

    if len(frame_steps) >= 2 and frame_steps[-1] == frame_steps[-2]:
        del frames[-2]
        del frame_steps[-2]
        del frame_times[-2]

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(frames[0], cmap=args.cmap, vmin=0, vmax=1, interpolation="nearest", animated=True)
    ax.set_title("kMC Occupancy")
    ax.set_xticks([])
    ax.set_yticks([])

    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", color="white", bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55})

    def update(frame_index):
        frame = frames[frame_index]
        image.set_data(frame)
        status.set_text(f"frame {frame_index + 1}/{len(frames)}\nstep {frame_steps[frame_index]}\ntime {frame_times[frame_index]:.2f}\ncoverage {float(frame.mean()):.3f}")
        return image, status

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / max(args.fps, 1), blit=True)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if out_path.suffix.lower() == ".gif":
            ani.save(out_path, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
        else:
            ani.save(out_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=args.bitrate), dpi=args.dpi)
        saved_path = out_path
    except Exception as exc:
        fallback = out_path.with_suffix(".gif")
        ani.save(fallback, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
        saved_path = fallback
        print(f"Fallback GIF saved due to {exc}.")    
    plt.close(fig)
    print("\nMovie saved:", saved_path, "Frames:", len(frames))

if __name__ == "__main__":
    main()
