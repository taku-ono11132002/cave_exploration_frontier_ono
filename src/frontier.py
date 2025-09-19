import numpy as np
def neighbors8(p,W,H):
    x,y=p
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0: continue
            nx,ny=x+dx,y+dy
            if 0<=nx<W and 0<=ny<H: yield (nx,ny)
def extract_frontiers(smap):
    free=smap.known_free_mask(); unk=smap.unknown_mask()
    H,W=free.shape; frontier=np.zeros_like(free,dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if not free[y,x]: continue
            for nx,ny in neighbors8((x,y),W,H):
                if unk[ny,nx]: frontier[y,x]=1; break
    return frontier
def cluster_frontiers(frontier_mask,min_size=5):
    H,W=frontier_mask.shape; visited=np.zeros_like(frontier_mask,dtype=np.uint8); clusters=[]
    for y in range(H):
        for x in range(W):
            if frontier_mask[y,x] and not visited[y,x]:
                stack=[(x,y)]; comp=[]; visited[y,x]=1
                while stack:
                    ux,uy=stack.pop(); comp.append((ux,uy))
                    for vx,vy in neighbors8((ux,uy),W,H):
                        if frontier_mask[vy,vx] and not visited[vy,vx]:
                            visited[vy,vx]=1; stack.append((vx,vy))
                if len(comp)>=min_size: clusters.append(comp)
    reps=[]
    for comp in clusters:
        xs=[p[0] for p in comp]; ys=[p[1] for p in comp]
        cx=int(round(np.mean(xs))); cy=int(round(np.mean(ys)))
        rep=min(comp,key=lambda p:(p[0]-cx)**2+(p[1]-cy)**2)
        reps.append((rep,comp))
    return reps
def estimate_unknown_mass(smap,rep_cell,radius=12):
    x0,y0=rep_cell; unk=smap.unknown_mask(); H,W=unk.shape
    ys=range(max(0,y0-radius),min(H,y0+radius+1))
    xs=range(max(0,x0-radius),min(W,x0+radius+1))
    c=0
    for y in ys:
        for x in xs:
            if (x-x0)**2+(y-y0)**2<=radius*radius and unk[y,x]: c+=1
    return float(c)
