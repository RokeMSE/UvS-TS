import numpy as np
from utils.zscore_normalize import zscore_exclude_index
def LSM(A_wave, u):
    A = A_wave.cpu().numpy()
    I = np.eye(A.shape[0])
    M1 = np.linalg.inv(I - A)  
    A_removed = A.copy()
    A_removed[u, :] = 0
    A_removed[:, u] = 0

    I = np.eye(A_removed.shape[0])
    M2 = np.linalg.inv(I - A_removed)

    M = M1 - M2
    impact_on_all_nodes = np.sum(M, axis=0)

    return zscore_exclude_index(impact_on_all_nodes, u)