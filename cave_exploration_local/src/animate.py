import matplotlib.pyplot as plt, matplotlib.animation as animation
from matplotlib.lines import Line2D
import numpy as np

def save_animation_mp4(frames,out_path):
    import matplotlib as mpl, imageio_ffmpeg
    mpl.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    N=len(frames[0].robots)
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=130)
    im=ax.imshow(frames[0].logodds,vmin=-6,vmax=6,cmap="gray_r",animated=True)
    scat_r=ax.scatter([],[],s=24)
    scat_f=ax.scatter([],[],marker="x",s=20)
    trails=[Line2D([],[],linewidth=0.9) for _ in range(N)]
    for ln in trails: ax.add_line(ln)
    trail_hist=[[] for _ in range(N)]
    text=ax.text(0.02,0.98,"",transform=ax.transAxes,va="top",ha="left",
                 fontsize=9,bbox=dict(facecolor="white",alpha=0.6,edgecolor="none"))
    ax.set_title("Frontier-best + self-best assignment"); ax.set_xlabel("x"); ax.set_ylabel("y")
    def update(i):
        fr=frames[i]; im.set_array(fr.logodds)
        scat_r.set_offsets(fr.robots[:,[0,1]] if len(fr.robots)>0 else np.empty((0,2)))
        scat_f.set_offsets(fr.reps[:,[0,1]]   if len(fr.reps)>0   else np.empty((0,2)))
        for rid in range(len(fr.robots)):
            trail_hist[rid].append((fr.robots[rid,0],fr.robots[rid,1]))
            xs=[p[0] for p in trail_hist[rid]]; ys=[p[1] for p in trail_hist[rid]]
            trails[rid].set_data(xs,ys)
        text.set_text(f"step: {fr.step}\nfrontiers: {fr.frontier_count}\nunknown: {fr.unknown_count}")
        return (im,scat_r,scat_f,*trails,text)
    ani=animation.FuncAnimation(fig,update,frames=len(frames),interval=80,blit=False)
    from matplotlib.animation import FFMpegWriter
    ani.save(out_path,writer=FFMpegWriter(fps=10),dpi=130); plt.close(fig)
def save_snapshot(frames,out_path):
    fr=frames[-1]; plt.figure(figsize=(5.5,5.5),dpi=130)
    plt.imshow(fr.logodds,vmin=-6,vmax=6,cmap="gray_r")
    if len(fr.robots)>0: plt.scatter(fr.robots[:,0],fr.robots[:,1],s=24)
    if len(fr.reps)>0: plt.scatter(fr.reps[:,0],fr.reps[:,1],s=20,marker="x")
    plt.title("Final Map"); plt.xlabel("x"); plt.ylabel("y")
    plt.savefig(out_path,bbox_inches="tight"); plt.close()
