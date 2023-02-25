from core.data_init_utils import *
from core.simulator import simulate
from model.utils import load_model
from numpy import seterr
seterr(all='raise')
# 8dB_50corr
# mean, std = 73.15985075872533, 8.892841565665687

# 8dB_10corr
# mean, std = 73.9509033483911, 9.951487711154964

# 8dB_5corr
# mean, std = 73.95431849519642, 9.95093532689939

# 8dBnoise
# mean, std = 73.56853596368727, 9.219616065619038

# 3dBnoise
# mean, std = 73.46376171091826, 5.514566913794413

def simulator_entry(hparams):
    if hparams.power_est_method == 'NN':
        model_path = 'model/large_channel_predict_runs/EDL/8dB_50corr'
        mean, std = 73.15985075872533, 8.892841565665687
        large_fading_NN = load_model(model_path, mean, std)
        hparams.large_fading_NN = large_fading_NN
    UE_list, BS_list, shadowFad_dB_map = init_all(hparams)
    init_step_idx = 0
    simulate(hparams, UE_list, BS_list, shadowFad_dB_map, init_step_idx)


if __name__ == '__main__':
    from config.hyperparam import hparams
    np.random.seed(hparams.seed)
    simulator_entry(hparams)


