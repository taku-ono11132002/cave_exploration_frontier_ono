import numpy as np # 数値計算ライブラリNumpyをインポート

def assign_frontier_best_then_self_best(C): # コスト行列Cに基づき、ロボットをフロンティアに割り当てる関数
    if C.size==0: return {} # コスト行列が空なら、空の割り当てを返す
    nR,nF=C.shape; assign={} # ロボット数(nR), フロンティア数(nF)を取得し、割り当て辞書を初期化
    # 各フロンティアに最良のドローンを確保
    for j in range(nF): # 全てのフロンティアについてループ
        i_best=int(np.argmin(C[:,j])); assign[i_best]=j # フロンティアjにとって最もコストが低いロボットi_bestを割り当てる
    # 残りのドローンは自分の最良へ（重複許容）
    for i in range(nR): # 全てのロボットについてループ
        if i not in assign: # まだ割り当てられていないロボットiについて
            j_best=int(np.argmin(C[i,:])); assign[i]=j_best # ロボットiにとって最もコストが低いフロンティアj_bestを割り当てる
    return assign # 最終的な割り当て結果を返す