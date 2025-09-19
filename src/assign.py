import numpy as np
def assign_frontier_best_then_self_best(C):
    if C.size==0: return {}
    nR,nF=C.shape; assign={}
    # 各フロンティアに最良のドローンを確保
    for j in range(nF):
        i_best=int(np.argmin(C[:,j])); assign[i_best]=j
    # 残りのドローンは自分の最良へ（重複許容）
    for i in range(nR):
        if i not in assign:
            j_best=int(np.argmin(C[i,:])); assign[i]=j_best
    return assign
