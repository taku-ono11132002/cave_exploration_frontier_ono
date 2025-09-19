import numpy as np, math # 数値計算ライブラリと数学ライブラリをインポート

class SLAMMap: # SLAM（自己位置推定と地図作成）で用いる占有格子地図を管理するクラス
    def __init__(self,H,W,L_free=-2.2,L_occ=2.2,L_min=-6,L_max=6,p_free_th=0.3,p_occ_th=0.7): # コンストラクタ
        self.logodds=np.zeros((H,W),dtype=np.float32) # 地図本体（log-odds表現）を0で初期化
        self.L_free=L_free; self.L_occ=L_occ; self.L_min=L_min; self.L_max=L_max # log-oddsの更新値と範囲
        self.p_free_th=p_free_th; self.p_occ_th=p_occ_th # 自由/占有/未知を判断するための確率の閾値
    def to_prob(self): return 1.0/(1.0+np.exp(-self.logodds)) # log-oddsを確率に変換するメソッド
    def known_free_mask(self): return self.to_prob()<self.p_free_th # 「既知の自由領域」のマスクを返すメソッド
    def unknown_mask(self): # 「未知領域」のマスクを返すメソッド
        p=self.to_prob(); return (p>=self.p_free_th)&(p<=self.p_occ_th) # 確率が自由とも占有とも言えない範囲

def _bresenham_line(x0,y0,x1,y1): # 2点間に直線を引くブレゼンハムのアルゴリズム
    pts=[]; dx=abs(x1-x0); sx=1 if x0<x1 else -1; dy=-abs(y1-y0); sy=1 if y0<y1 else -1; err=dx+dy; x,y=x0,y0 # アルゴリズムの変数を初期化
    while True: # 終点に達するまでループ
        pts.append((x,y)) # 現在の点をリストに追加
        if x==x1 and y==y1: break # 終点に達したら終了
        e2=2*err # エラー項を計算
        if e2>=dy: err+=dy; x+=sx # x方向に移動
        if e2<=dx: err+=dx; y+=sy # y方向に移動
    return pts # 直線上の点のリストを返す

def sensor_update_lidar(smap,true_occ,cx,cy,max_range=24,fov=2*math.pi,n_beams=90): # LIDARセンサーの観測に基づいてSLAMマップを更新する関数
    H,W=true_occ.shape; x0,y0=int(cx),int(cy) # 地図サイズとセンサーの整数座標を取得
    if not (0<=x0<W and 0<=y0<H): return # センサーが地図範囲外なら何もしない
    angles=np.linspace(-fov/2,+fov/2,n_beams) # 指定された視野角とビーム数で各ビームの角度を計算
    for th in angles: # 各ビームについてループ
        x1=int(round(x0+max_range*math.cos(th))); y1=int(round(y0+max_range*math.sin(th))) # ビームの最大到達点を計算
        x1=min(max(0,x1),W-1); y1=min(max(0,y1),H-1) # 到達点を地図範囲内に収める
        line=_bresenham_line(x0,y0,x1,y1); hit=None # センサーから到達点までの直線上のセルを計算
        for ux,uy in line: # 直線上の各セルをチェック
            if true_occ[uy,ux]==1: hit=(ux,uy); break # 障害物に当たったら、その位置を記録してループを抜ける
        if hit is None: # 障害物に当たらなかった場合
            for ux,uy in line: # 直線上の全てのセルについて
                smap.logodds[uy,ux]=np.clip(smap.logodds[uy,ux]+smap.L_free,smap.L_min,smap.L_max) # 「自由空間」としてlog-oddsを更新
        else: # 障害物に当たった場合
            for ux,uy in line: # センサーから衝突点までの各セルについて
                if (ux,uy)==hit: break # 衝突点に達したらループを抜ける
                smap.logodds[uy,ux]=np.clip(smap.logodds[uy,ux]+smap.L_free,smap.L_min,smap.L_max) # 途中のセルを「自由空間」として更新
            hx,hy=hit; smap.logodds[hy,hx]=np.clip(smap.logodds[hy,hx]+smap.L_occ,smap.L_min,smap.L_max) # 衝突したセルを「占有空間」として更新