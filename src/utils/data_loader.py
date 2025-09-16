# Use PyTorch DataLoader
import pickle
import numpy as np

def load_data_PEMS_BAY(file_path):

    X = np.load(file_path + "/PEMSBAY.npy").transpose((1, 2, 0))
    # (N, F, T)

    try:
        with open(file_path + "/adj_mx_bay.pkl", 'rb') as f:
            A = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file_path + "/adj_mx_bay.pkl", 'rb') as f:
            A = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file_path + "/adj_mx_bay.pkl", ':', e)
        raise

    X = X.astype(np.double)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A[2], X, means, stds