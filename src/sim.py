import numpy as np, math # 数値計算ライブラリと数学ライブラリをインポート
from dataclasses import dataclass # データクラスをインポート
from typing import Tuple, Optional # 型ヒント用のクラスをインポート
from .slam import SLAMMap, sensor_update_lidar # slamモジュールからクラスと関数をインポート
from .frontier import extract_frontiers, cluster_frontiers, estimate_unknown_mass # frontierモジュールから関数をインポート
from .planner import build_cost_matrix, neighbors4, manhattan # plannerモジュールから関数をインポート
from .assign import assign_frontier_best_then_self_best # assignモジュールから関数をインポート

@dataclass
class RobotState: # ロボットの状態を保持するデータクラス
    rid:int; pos:Tuple[int,int]; goal:Optional[Tuple[int,int]]=None; goal_id:Optional[int]=None; last_assign_t:float=-1e9

@dataclass
class Frame: # アニメーションの1フレームの状態を保持するデータクラス
    logodds: np.ndarray; robots: np.ndarray; reps: np.ndarray; step: int; frontier_count: int; unknown_count: int

def make_y_cave_down(W=120, H=120, corridor=6, length=50): # Y字型の洞窟環境を生成する関数
    """
    下部から上部中央まで幹を伸ばし、そこから左下・右下に枝分かれするY字洞窟（自由=0, 占有=1）を生成する。
    """
    occ = np.ones((H, W), dtype=np.uint8) # マップを全て占有（通行不可）で初期化
    cx = W // 2; junc_y = H - 45 # 洞窟の中心x座標と分岐点のy座標
    for y in range(8, junc_y + 1): # Y字の幹の部分を作成
        occ[y, max(0, cx-corridor//2):min(W, cx+corridor//2+1)] = 0 # 幹の部分を通行可能にする
    for i in range(length): # Y字の分岐部分を作成
        bx,by=cx-i,junc_y+i # 左下の枝の座標
        if 0<=bx<W and 0<=by<H: # 範囲内なら
            occ[max(0,by-corridor//2):min(H,by+corridor//2+1),
                max(0,bx-corridor//2):min(W,bx+corridor//2+1)] = 0 # 左下の枝を通行可能にする
        bx2,by2=cx+i,junc_y+i # 右下の枝の座標
        if 0<=bx2<W and 0<=by2<H: # 範囲内なら
            occ[max(0,by2-corridor//2):min(H,by2+corridor//2+1),
                max(0,bx2-corridor//2):min(W,bx2+corridor//2+1)] = 0 # 右下の枝を通行可能にする
    return occ # 生成した洞窟マップを返す

def run_simulation_custom(seed=0): # シミュレーションを実行するメイン関数
    np.random.seed(seed) # 乱数シードを設定して再現性を確保
    W,H=120,120 # マップの幅と高さを設定
    true_occ=make_y_cave_down(W,H) # Y字洞窟の真のマップ（答え）を生成
    smap=SLAMMap(H,W) # ロボットが作成するSLAMマップを初期化
    robots=[RobotState(0,(W//2 - 4, 12)), # ロボットの初期状態をリストとして設定
            RobotState(1,(W//2 + 4, 12)),
            RobotState(2,(W//2,     18))]
    frames=[]; t=0.0; last_periodic=-1e9; period=6.0; cooldown=2.5; eta=0.15 # シミュレーション用の変数を初期化
    for k in range(300): # メインループ（最大300ステップ）
        # sensing: センサーによる観測フェーズ
        for r in robots: # 各ロボットについて
            sensor_update_lidar(smap,true_occ,r.pos[0],r.pos[1]) # LIDARで観測し、SLAMマップを更新
        # frontiers: フロンティアの計算フェーズ
        frontier_mask=extract_frontiers(smap) # SLAMマップからフロンティアを抽出
        reps=cluster_frontiers(frontier_mask, min_size=5) # フロンティアをクラスタリング
        frontier_reps=[rep for (rep,_) in reps] # クラスタから代表点のリストを作成
        U_list=[estimate_unknown_mass(smap, rep) for rep,_ in reps] # 各フロンティアの未知領域の質量を推定
        # costs: コスト計算フェーズ
        known_free=smap.known_free_mask() # 現在の既知の自由領域を取得
        rob_pos=[r.pos for r in robots] # 全ロボットの現在位置リスト
        C=build_cost_matrix(rob_pos,reps,U_list,known_free,W,H) if frontier_reps else np.zeros((len(robots),0)) # コスト行列を計算
        # reassign?: 再割り当て判断フェーズ
        def need_reassign(t): # 再割り当てが必要か判断する内部関数
            nonlocal last_periodic # 外側の変数を参照
            if (t-last_periodic)>=period: return True # 定期的な再割り当て時期か
            if len(frontier_reps)==0 or C.size==0: return False # フロンティアがなければ不要
            for i,r in enumerate(robots): # 各ロボットについて
                if r.goal_id is None: return True # 目標がなければ再割り当て
                if (t-r.last_assign_t)<cooldown: continue # クールダウン中ならスキップ
                j_curr=r.goal_id # 現在の目標ID
                if not (0<=j_curr<C.shape[1]): return True # 現在の目標が無効なら再割り当て
                c_curr=C[i,j_curr]; j_best=int(np.argmin(C[i,:])); c_best=C[i,j_best] # 現在のコストと最良コストを比較
                if c_best<=(1.0-eta)*c_curr: return True # より良い目標が見つかれば再割り当て
            return False # 上記以外は不要
        if need_reassign(t): # 再割り当てが必要な場合
            if frontier_reps: # フロンティアがあれば
                assign=assign_frontier_best_then_self_best(C) # 割り当てアルゴリズムを実行
                for i,r in enumerate(robots): # 各ロボットについて
                    if i in assign: # 新しい割り当てがあれば
                        j=assign[i]; r.goal=frontier_reps[j]; r.goal_id=j; r.last_assign_t=t # ロボットの目標を更新
            last_periodic=t # 定期再割り当ての時刻を更新
        # move: 移動フェーズ
        for r in robots: # 各ロボットについて
            if r.goal is None: continue # 目標がなければ移動しない
            gx,gy=r.goal # 目標座標
            cands=[v for v in neighbors4(r.pos,W,H) if known_free[v[1],v[0]]] # 通行可能な隣接セルを候補に
            if cands: # 候補があれば
                best=min(cands,key=lambda v:manhattan(v,(gx,gy))) # 目標に最も近づくセルを選択
                if manhattan(best,(gx,gy))<manhattan(r.pos,(gx,gy)): # 目標に近づく場合のみ
                    r.pos=best # 位置を更新
        unknown_cnt=int((np.abs(smap.logodds)<0.5).sum()) # 未知領域のセル数をカウント
        frames.append(Frame(smap.logodds.copy(),np.array([r.pos for r in robots],int), # 現在の状態をフレームとして保存
                            np.array(frontier_reps,int) if frontier_reps else np.zeros((0,2),int),
                            k,len(frontier_reps),unknown_cnt))
        # finish: 終了条件のチェック
        if len(frontier_reps)==0 and k>20: break # 探査可能なフロンティアがなくなり、一定時間経過したら終了
        t+=0.5 # シミュレーション時刻を更新
    return frames # 全フレームのリストを返す