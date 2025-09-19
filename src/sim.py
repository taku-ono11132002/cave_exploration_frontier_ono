import numpy as np, math
from dataclasses import dataclass
from typing import Tuple, Optional
from .slam import SLAMMap, sensor_update_lidar
from .frontier import extract_frontiers, cluster_frontiers, estimate_unknown_mass
from .planner import build_cost_matrix, neighbors4, manhattan
from .assign import assign_frontier_best_then_self_best

@dataclass
class RobotState:
    rid:int; pos:Tuple[int,int]; goal:Optional[Tuple[int,int]]=None; goal_id:Optional[int]=None; last_assign_t:float=-1e9
@dataclass
class Frame:
    logodds: np.ndarray; robots: np.ndarray; reps: np.ndarray; step: int; frontier_count: int; unknown_count: int

def make_y_cave_down(W=120, H=120, corridor=6, length=50):
    """
    下部から上部中央まで幹を伸ばし、そこから左下・右下に枝分かれするY字洞窟（自由=0, 占有=1）を生成する。
    """
    occ = np.ones((H, W), dtype=np.uint8)
    cx = W // 2; junc_y = H - 45
    for y in range(8, junc_y + 1):
        occ[y, max(0, cx-corridor//2):min(W, cx+corridor//2+1)] = 0
    for i in range(length):
        bx,by=cx-i,junc_y+i
        if 0<=bx<W and 0<=by<H:
            occ[max(0,by-corridor//2):min(H,by+corridor//2+1),
                max(0,bx-corridor//2):min(W,bx+corridor//2+1)] = 0
        bx2,by2=cx+i,junc_y+i
        if 0<=bx2<W and 0<=by2<H:
            occ[max(0,by2-corridor//2):min(H,by2+corridor//2+1),
                max(0,bx2-corridor//2):min(W,bx2+corridor//2+1)] = 0
    return occ

def run_simulation_custom(seed=0):
    np.random.seed(seed)
    W,H=120,120
    true_occ=make_y_cave_down(W,H)
    smap=SLAMMap(H,W)
    robots=[RobotState(0,(W//2 - 4, 12)),
            RobotState(1,(W//2 + 4, 12)),
            RobotState(2,(W//2,     18))]
    frames=[]; t=0.0; last_periodic=-1e9; period=6.0; cooldown=2.5; eta=0.15
    for k in range(300):
        # sensing
        for r in robots:
            sensor_update_lidar(smap,true_occ,r.pos[0],r.pos[1])
        # frontiers
        frontier_mask=extract_frontiers(smap)
        reps=cluster_frontiers(frontier_mask, min_size=5)
        frontier_reps=[rep for (rep,_) in reps]
        U_list=[estimate_unknown_mass(smap, rep) for rep,_ in reps]
        # costs
        known_free=smap.known_free_mask()
        rob_pos=[r.pos for r in robots]
        C=build_cost_matrix(rob_pos,reps,U_list,known_free,W,H) if frontier_reps else np.zeros((len(robots),0))
        # reassign?
        def need_reassign(t):
            nonlocal last_periodic
            if (t-last_periodic)>=period: return True
            if len(frontier_reps)==0 or C.size==0: return False
            for i,r in enumerate(robots):
                if r.goal_id is None: return True
                if (t-r.last_assign_t)<cooldown: continue
                j_curr=r.goal_id
                if not (0<=j_curr<C.shape[1]): return True
                c_curr=C[i,j_curr]; j_best=int(np.argmin(C[i,:])); c_best=C[i,j_best]
                if c_best<=(1.0-eta)*c_curr: return True
            return False
        if need_reassign(t):
            if frontier_reps:
                assign=assign_frontier_best_then_self_best(C)
                for i,r in enumerate(robots):
                    if i in assign:
                        j=assign[i]; r.goal=frontier_reps[j]; r.goal_id=j; r.last_assign_t=t
            last_periodic=t
        # move
        for r in robots:
            if r.goal is None: continue
            gx,gy=r.goal
            cands=[v for v in neighbors4(r.pos,W,H) if known_free[v[1],v[0]]]
            if cands:
                best=min(cands,key=lambda v:manhattan(v,(gx,gy)))
                if manhattan(best,(gx,gy))<manhattan(r.pos,(gx,gy)):
                    r.pos=best
        unknown_cnt=int((np.abs(smap.logodds)<0.5).sum())
        frames.append(Frame(smap.logodds.copy(),np.array([r.pos for r in robots],int),
                            np.array(frontier_reps,int) if frontier_reps else np.zeros((0,2),int),
                            k,len(frontier_reps),unknown_cnt))
        # finish: reachable frontiers exhausted
        if len(frontier_reps)==0 and k>20: break
        t+=0.5
    return frames
