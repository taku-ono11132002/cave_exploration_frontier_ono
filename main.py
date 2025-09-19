import os
from src.sim import run_simulation_custom
from src.animate import save_animation_mp4, save_snapshot
from datetime import datetime

def main():
    frames = run_simulation_custom(seed=0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, f"y_cave_exploration_{stamp}.mp4")
    png_path = os.path.join(out_dir, f"y_cave_final_{stamp}.png")
    save_animation_mp4(frames, mp4_path)
    save_snapshot(frames, png_path)
    print("Saved:", mp4_path, png_path)

if __name__ == "__main__":
    main()
