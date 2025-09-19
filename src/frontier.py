import numpy as np # 数値計算ライブラリNumpyをインポート

def neighbors8(p,W,H): # ある点pの周囲8方向の隣接点の座標を返すジェネレータ
    x,y=p # 点pの座標を取得
    for dx in (-1,0,1): # x方向の移動量
        for dy in (-1,0,1): # y方向の移動量
            if dx==0 and dy==0: continue # 中心点はスキップ
            nx,ny=x+dx,y+dy # 隣接点の座標を計算
            if 0<=nx<W and 0<=ny<H: yield (nx,ny) # 地図の範囲内なら座標を返す

def extract_frontiers(smap): # SLAMマップからフロンティア（未知と既知の境界）を抽出する関数
    free=smap.known_free_mask(); unk=smap.unknown_mask() # 「既知の自由領域」と「未知領域」のマスクを取得
    H,W=free.shape; frontier=np.zeros_like(free,dtype=np.uint8) # 地図のサイズを取得し、フロンティアマスクを初期化
    for y in range(H): # 地図の各行についてループ
        for x in range(W): # 地図の各列についてループ
            if not free[y,x]: continue # 現在のセルが自由領域でなければスキップ
            for nx,ny in neighbors8((x,y),W,H): # 周囲8方向の隣接セルをチェック
                if unk[ny,nx]: frontier[y,x]=1; break # 隣接セルが未知領域なら、現在のセルをフロンティアとしてマークし、ループを抜ける
    return frontier # フロンティアマスクを返す

def cluster_frontiers(frontier_mask,min_size=5): # フロンティアセルをクラスタリング（グループ化）する関数
    H,W=frontier_mask.shape; visited=np.zeros_like(frontier_mask,dtype=np.uint8); clusters=[] # サイズ、訪問済みマスク、クラスタリストを初期化
    for y in range(H): # 地図の各行についてループ
        for x in range(W): # 地図の各列についてループ
            if frontier_mask[y,x] and not visited[y,x]: # 未訪問のフロンティアセルが見つかったら
                stack=[(x,y)]; comp=[]; visited[y,x]=1 # 新しいクラスタの探索を開始（深さ優先探索）
                while stack: # スタックが空になるまで
                    ux,uy=stack.pop(); comp.append((ux,uy)) # セルを取り出し、現在のクラスタに追加
                    for vx,vy in neighbors8((ux,uy),W,H): # 隣接セルをチェック
                        if frontier_mask[vy,vx] and not visited[vy,vx]: # 未訪問のフロンティアセルなら
                            visited[vy,vx]=1; stack.append((vx,vy)) # 訪問済みにし、スタックに追加
                if len(comp)>=min_size: clusters.append(comp) # クラスタのサイズが最小サイズ以上ならリストに追加
    reps=[] # 各クラстаの代表点を格納するリスト
    for comp in clusters: # 各クラスタについてループ
        xs=[p[0] for p in comp]; ys=[p[1] for p in comp] # クラスタ内の全セルのx, y座標を取得
        cx=int(round(np.mean(xs))); cy=int(round(np.mean(ys))) # クラスタの重心を計算
        rep=min(comp,key=lambda p:(p[0]-cx)**2+(p[1]-cy)**2) # 重心に最も近いセルを代表点として選択
        reps.append((rep,comp)) # (代表点, クラスタ)のタプルをリストに追加
    return reps # 代表点とクラスタのリストを返す

def estimate_unknown_mass(smap,rep_cell,radius=12): # フロンティア代表点の周囲にある未知領域の「質量」（セル数）を推定する関数
    x0,y0=rep_cell; unk=smap.unknown_mask(); H,W=unk.shape # 代表点座標、未知領域マスク、地図サイズを取得
    ys=range(max(0,y0-radius),min(H,y0+radius+1)) # 探索範囲のy座標
    xs=range(max(0,x0-radius),min(W,x0+radius+1)) # 探索範囲のx座標
    c=0 # 未知セル数のカウンタ
    for y in ys: # 探索範囲の各行についてループ
        for x in xs: # 探索範囲の各列についてループ
            if (x-x0)**2+(y-y0)**2<=radius*radius and unk[y,x]: c+=1 # 半径内の円にあり、かつ未知領域ならカウントアップ
    return float(c) # 未知セル数を返す