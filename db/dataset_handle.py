import numpy as np

def prepare_sequential_dataset(file_path, obs_len, pred_len):
    data = np.load(file_path, allow_pickle=True)
    X = []
    y = []
    for idx in range(data.shape[1]-obs_len-pred_len):
        _X = data[:, idx:idx+obs_len]
        _y = data[:, idx+obs_len:idx+obs_len+pred_len]
        X.append(_X)
        y.append(_y)

    return np.array(X), np.array(y)

if __name__ == '__main__':
    file_path = 'large_fading_dB_data.npy'
    obs_len = 5
    pred_len = 1
    X, y = prepare_sequential_dataset(file_path, obs_len, pred_len)
    X = np.reshape(X, (-1, obs_len))
    y = np.reshape(y, (-1, pred_len))
    np.save('large_fading_dB_data.npy', {'x':X, 'y':y}, allow_pickle=True)
