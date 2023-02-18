from core.data_init_utils import *
from core.simulator import simulate

def simulator_entry(hparams):
    UE_list, BS_list, shadowFad_dB_map = init_all(hparams)
    init_step_idx = 0
    simulate(hparams, UE_list, BS_list, shadowFad_dB_map, init_step_idx)
    pass



if __name__ == '__main__':
    from config.hyperparam import hparams
    simulator_entry(hparams)


