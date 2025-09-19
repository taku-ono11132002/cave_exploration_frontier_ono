import numpy as np
def neighbors4(p,W,H):
    x,y=p
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx,ny=x+dx,y+dy
        if 0<=nx<W and 0<=ny<H: yield (nx,ny)
def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
def astar_length(start,goal,known_free,W,H):
    if start==goal: return 0.0
    sx,sy=start; gx,gy=goal
    if not (0<=gx<W and 0<=gy<H and 0<=sx<W and 0<=sy<H): return float('inf')
    if not known_free[gy,gx] or not known_free[sy,sx]: return float('inf')
    import heapq
    openh=[]; heapq.heappush(openh,(manhattan(start,goal),0.0,start))
    g={start:0.0}; closed=set()
    while openh:
        f, gc, u=heapq.heappop(openh)
        if u in closed: continue
        if u==goal: return gc
        closed.add(u)
        for v in neighbors4(u,W,H):
            if not known_free[v[1],v[0]]: continue
            ng=gc+1.0
            if ng<g.get(v,1e18):
                g[v]=ng; heapq.heappush(openh,(ng+manhattan(v,goal),ng,v))
    return float('inf')
def build_cost_matrix(robots_pos,reps,U_list,known_free,W,H):
    d=np.zeros((len(robots_pos),len(reps)),float)
    for i,rpos in enumerate(robots_pos):
        for j,(rep,comp) in enumerate(reps):
            d[i,j]=astar_length(rpos,rep,known_free,W,H)
    maxd=np.max(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else 1.0
    if maxd<=1e-9: maxd=1.0
    maxU=max(U_list) if len(U_list)>0 else 1.0
    if maxU<=1e-9: maxU=1.0
    C=np.full_like(d,1e6,float)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if np.isfinite(d[i,j]):
                C[i,j]=(d[i,j]/maxd)-(U_list[j]/maxU)
    return C
