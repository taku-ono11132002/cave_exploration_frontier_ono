import numpy as np, math
class SLAMMap:
    def __init__(self,H,W,L_free=-2.2,L_occ=2.2,L_min=-6,L_max=6,p_free_th=0.3,p_occ_th=0.7):
        self.logodds=np.zeros((H,W),dtype=np.float32)
        self.L_free=L_free; self.L_occ=L_occ; self.L_min=L_min; self.L_max=L_max
        self.p_free_th=p_free_th; self.p_occ_th=p_occ_th
    def to_prob(self): return 1.0/(1.0+np.exp(-self.logodds))
    def known_free_mask(self): return self.to_prob()<self.p_free_th
    def unknown_mask(self):
        p=self.to_prob(); return (p>=self.p_free_th)&(p<=self.p_occ_th)

def _bresenham_line(x0,y0,x1,y1):
    pts=[]; dx=abs(x1-x0); sx=1 if x0<x1 else -1; dy=-abs(y1-y0); sy=1 if y0<y1 else -1; err=dx+dy; x,y=x0,y0
    while True:
        pts.append((x,y))
        if x==x1 and y==y1: break
        e2=2*err
        if e2>=dy: err+=dy; x+=sx
        if e2<=dx: err+=dx; y+=sy
    return pts

def sensor_update_lidar(smap,true_occ,cx,cy,max_range=24,fov=2*math.pi,n_beams=90):
    H,W=true_occ.shape; x0,y0=int(cx),int(cy)
    if not (0<=x0<W and 0<=y0<H): return
    angles=np.linspace(-fov/2,+fov/2,n_beams)
    for th in angles:
        x1=int(round(x0+max_range*math.cos(th))); y1=int(round(y0+max_range*math.sin(th)))
        x1=min(max(0,x1),W-1); y1=min(max(0,y1),H-1)
        line=_bresenham_line(x0,y0,x1,y1); hit=None
        for ux,uy in line:
            if true_occ[uy,ux]==1: hit=(ux,uy); break
        if hit is None:
            for ux,uy in line:
                smap.logodds[uy,ux]=np.clip(smap.logodds[uy,ux]+smap.L_free,smap.L_min,smap.L_max)
        else:
            for ux,uy in line:
                if (ux,uy)==hit: break
                smap.logodds[uy,ux]=np.clip(smap.logodds[uy,ux]+smap.L_free,smap.L_min,smap.L_max)
            hx,hy=hit; smap.logodds[hy,hx]=np.clip(smap.logodds[hy,hx]+smap.L_occ,smap.L_min,smap.L_max)
