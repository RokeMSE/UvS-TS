import numpy as np
from utils.zscore_normalize import zscore_exclude_index

def compute_irf(A_comp, H):
    irfs = []
    A_power = np.eye(A_comp.shape[0])
    
    for h in range(H):
        irfs.append(A_power.copy())
        A_power = A_power @ A_comp
        
    return irfs

def get_irf_node(i, j, irfs, T, N, F):
    # i: source node
    NF = N *F
    # j: target node
    res = []
    
    for Psi in irfs:
        # chỉ lấy phần NF đầu
        Psi0 = Psi[:NF, :NF]
        
        val = Psi0[j*F:(j+1)*F, i*F:(i+1)*F].sum()
        res.append(val)
        
    return np.array(res)

def VAR_IRF(X, u, p=2, H=10):
    '''
        X: NxFxT (original data)
    '''
    X = X.transpose(2, 0, 1)
    T, N, F = X.shape
    X_flat = X.reshape(T, N * F)
    X_diff = X_flat[1:] - X_flat[:-1]
    Y = X_diff[p:]  # target

    X_lag = []
    for i in range(1, p+1):
        X_lag.append(X_diff[p-i: -i])

    X_lag = np.concatenate(X_lag, axis=1) 
    A_hat = np.linalg.pinv(X_lag) @ Y
    NF = N * F
    A_list = []

    for i in range(p):
        A_i = A_hat[i*NF:(i+1)*NF, :]
        A_list.append(A_i)

    A_comp = np.zeros((p*NF, p*NF))
    A_comp[:NF, :] = np.concatenate(A_list, axis=0).T

    for i in range(1, p):
        A_comp[i*NF:(i+1)*NF, (i-1)*NF:i*NF] = np.eye(NF)

    irfs = compute_irf(A_comp, H=10)
    impact_list = []
    for j in range(N):
        irf = get_irf_node(u, j, irfs, T, N, F)
        
        score = irf.sum()
        impact_list.append(score)

    return zscore_exclude_index(impact_list, u)