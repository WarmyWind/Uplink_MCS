import numpy as np
import scipy.io as scio

def prepare_sequential_dataset(file_path, obs_len, pred_len, norm=True):
    data = np.load(file_path, allow_pickle=True)
    if norm:
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
    X = []
    y = []
    for idx in range(data.shape[1]-obs_len-pred_len):
        _X = data[:, idx:idx+obs_len]
        _y = data[:, idx+obs_len:idx+obs_len+pred_len]
        X.append(_X)
        y.append(_y)

    return np.array(X), np.array(y)

def mat_to_npy(file_path, index):
    mat = scio.loadmat(file_path)
    data = mat.get(index)
    return np.array(data)

if __name__ == '__main__':
    file_path = '8dB_50corr_large_fading_dB_data.npy'

    obs_len = 5
    pred_len = 1
    X, y = prepare_sequential_dataset(file_path, obs_len, pred_len, norm=True)
    X = np.reshape(X, (-1, obs_len))
    y = np.reshape(y, (-1, pred_len))
    np.save('normed_8dB_50corr_large_fading_dB_dataset.npy', {'x':X, 'y':y}, allow_pickle=True)


