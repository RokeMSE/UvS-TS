import math
import numpy as np

def zscore_exclude_index(lst, u):
    vals = [x for i, x in enumerate(lst) if i != u]
    
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
    
    if std == 0:
        lst[u] = 0
        max_val = np.max(np.abs(lst))

        if max_val > 0:
            norm_impacts = lst / max_val
        else:
            norm_impacts = lst
        lst[u] = 9999
        return lst
    
    result = []
    for i, x in enumerate(lst):
        if i == u:
            result.append(9999)
        else:
            z = (x - mean) / std
            result.append(z)
    
    return result