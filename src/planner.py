import numpy as np # 数値計算ライブラリNumpyをインポート

def neighbors4(p,W,H): # ある点pの上下左右4方向の隣接点の座標を返すジェネレータ
    x,y=p # 点pの座標を取得
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)): # 上下左右の移動量
        nx,ny=x+dx,y+dy # 隣接点の座標を計算
        if 0<=nx<W and 0<=ny<H: yield (nx,ny) # 地図の範囲内なら座標を返す

def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1]) # 2点間のマンハッタン距離を計算する関数

def astar_length(start,goal,known_free,W,H): # A*アルゴリズムでstartからgoalまでの最短経路長を計算する関数
    if start==goal: return 0.0 # スタートとゴールが同じなら距離は0
    sx,sy=start; gx,gy=goal # スタートとゴールの座標を取得
    if not (0<=gx<W and 0<=gy<H and 0<=sx<W and 0<=sy<H): return float('inf') # 範囲外なら到達不能
    if not known_free[gy,gx] or not known_free[sy,sx]: return float('inf') # 通行不能なら到達不能
    import heapq # 優先度付きキュー（ヒープ）をインポート
    openh=[]; heapq.heappush(openh,(manhattan(start,goal),0.0,start)) # オープンリスト（ヒープ）を初期化し、スタート地点を追加
    g={start:0.0}; closed=set() # スタートからのコストgと、探索済みリストclosedを初期化
    while openh: # オープンリストが空になるまで探索
        f, gc, u=heapq.heappop(openh) # f値が最小のノードuを取り出す
        if u in closed: continue # 探索済みならスキップ
        if u==goal: return gc # ゴールに到達したらコストgcを返す
        closed.add(u) # ノードuを探索済みに追加
        for v in neighbors4(u,W,H): # 隣接ノードvをチェック
            if not known_free[v[1],v[0]]: continue # 通行不能ならスキップ
            ng=gc+1.0 # u経由のvまでの新しいコスト
            if ng<g.get(v,1e18): # 既存のコストより新コストが小さければ
                g[v]=ng; heapq.heappush(openh,(ng+manhattan(v,goal),ng,v)) # コストを更新し、vをオープンリストに追加
    return float('inf') # ゴールに到達できなければ無限大を返す

def build_cost_matrix(robots_pos,reps,U_list,known_free,W,H): # ロボットとフロンティア間のコスト行列を構築する関数
    d=np.zeros((len(robots_pos),len(reps)),float) # 距離行列dを初期化
    for i,rpos in enumerate(robots_pos): # 各ロボットについてループ
        for j,(rep,comp) in enumerate(reps): # 各フロンティアについてループ
            d[i,j]=astar_length(rpos,rep,known_free,W,H) # A*でロボットiからフロンティアjまでの経路長を計算
    maxd=np.max(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else 1.0 # 距離の最大値を取得（正規化のため）
    if maxd<=1e-9: maxd=1.0 # 0除算を避ける
    maxU=max(U_list) if len(U_list)>0 else 1.0 # 未知領域の質量の最大値を取得（正規化のため）
    if maxU<=1e-9: maxU=1.0 # 0除算を避ける
    C=np.full_like(d,1e6,float) # コスト行列Cを大きな値で初期化
    for i in range(d.shape[0]): # 行列の各行（ロボット）についてループ
        for j in range(d.shape[1]): # 行列の各列（フロンティア）についてループ
            if np.isfinite(d[i,j]): # 経路が存在する場合
                C[i,j]=(d[i,j]/maxd)-(U_list[j]/maxU) # コストを計算（正規化距離 - 正規化未知質量）
    return C # コスト行列を返す