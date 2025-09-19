import matplotlib.pyplot as plt, matplotlib.animation as animation # 描画とアニメーション作成のためのライブラリをインポート
from matplotlib.lines import Line2D # 2Dの線を描画するためのクラスをインポート
import numpy as np # 数値計算ライブラリNumpyをインポート

def save_animation_mp4(frames,out_path): # シミュレーションの各フレームからMP4アニメーションを保存する関数
    import matplotlib as mpl, imageio_ffmpeg # matplotlibとffmpegラッパーをインポート
    mpl.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe() # matplotlibにffmpegのパスを設定
    N=len(frames[0].robots) # 最初のフレームからロボットの数を取得
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=130) # 描画用の図と軸を作成
    im=ax.imshow(frames[0].logodds,vmin=-6,vmax=6,cmap="gray_r",animated=True) # 地図（logodds）を画像として表示
    scat_r=ax.scatter([],[],s=24) # ロボットの位置を示すための散布図オブジェクト
    scat_f=ax.scatter([],[],marker="x",s=20) # フロンティアの位置を示すための散布図オブジェクト
    trails=[Line2D([],[],linewidth=0.9) for _ in range(N)] # 各ロボットの軌跡を描画するためのLine2Dオブジェクトのリスト
    for ln in trails: ax.add_line(ln) # 軌跡オブジェクトを軸に追加
    trail_hist=[[] for _ in range(N)] # 各ロボットの過去の位置履歴を保存するリスト
    text=ax.text(0.02,0.98,"",transform=ax.transAxes,va="top",ha="left", # テキスト表示用のオブジェクト
                 fontsize=9,bbox=dict(facecolor="white",alpha=0.6,edgecolor="none"))
    ax.set_title("Frontier-best + self-best assignment"); ax.set_xlabel("x"); ax.set_ylabel("y") # グラフのタイトルと軸ラベルを設定
    def update(i): # アニメーションの各フレームを更新する関数
        fr=frames[i]; im.set_array(fr.logodds) # i番目のフレームを取得し、地図画像を更新
        scat_r.set_offsets(fr.robots[:,[0,1]] if len(fr.robots)>0 else np.empty((0,2))) # ロボットの散布図の位置を更新
        scat_f.set_offsets(fr.reps[:,[0,1]]   if len(fr.reps)>0   else np.empty((0,2))) # フロンティアの散布図の位置を更新
        for rid in range(len(fr.robots)): # 各ロボットについてループ
            trail_hist[rid].append((fr.robots[rid,0],fr.robots[rid,1])) # 現在位置を軌跡履歴に追加
            xs=[p[0] for p in trail_hist[rid]]; ys=[p[1] for p in trail_hist[rid]] # 履歴からx, y座標のリストを作成
            trails[rid].set_data(xs,ys) # 軌跡のデータを更新
        text.set_text(f"step: {fr.step}\nfrontiers: {fr.frontier_count}\nunknown: {fr.unknown_count}") # テキスト情報を更新
        return (im,scat_r,scat_f,*trails,text) # 更新された描画オブジェクトを返す
    ani=animation.FuncAnimation(fig,update,frames=len(frames),interval=80,blit=False) # アニメーションオブジェクトを作成
    from matplotlib.animation import FFMpegWriter # 動画書き出し用のライターをインポート
    ani.save(out_path,writer=FFMpegWriter(fps=10),dpi=130); plt.close(fig) # アニメーションを指定されたパスに保存し、図を閉じる

def save_snapshot(frames,out_path): # シミュレーションの最終状態を画像として保存する関数
    fr=frames[-1]; plt.figure(figsize=(5.5,5.5),dpi=130) # 最終フレームを取得し、新しい図を作成
    plt.imshow(fr.logodds,vmin=-6,vmax=6,cmap="gray_r") # 最終的な地図を表示
    if len(fr.robots)>0: plt.scatter(fr.robots[:,0],fr.robots[:,1],s=24) # ロボットの最終位置をプロット
    if len(fr.reps)>0: plt.scatter(fr.reps[:,0],fr.reps[:,1],s=20,marker="x") # フロンティアの最終位置をプロット
    plt.title("Final Map"); plt.xlabel("x"); plt.ylabel("y") # グラフのタイトルと軸ラベルを設定
    plt.savefig(out_path,bbox_inches="tight"); plt.close() # 画像を指定されたパスに保存し、図を閉じる