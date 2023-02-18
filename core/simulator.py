from core.info_update import update_posi, update_CSI, init_access


def _step(hparam, UE_list, BS_list, shadowFad_dB_map, step_idx):
    update_posi(UE_list, step_idx)
    update_CSI(hparam, UE_list, BS_list, shadowFad_dB_map)

    est_SINR()
    cal_spectral_efficiency()
    cal_true_SINR()
    cal_spectral_efficiency()
    return


def simulate(hparam, UE_list, BS_list, shadowFad_dB_map, init_step_idx):
    init_access(hparam, UE_list, BS_list, shadowFad_dB_map, init_step_idx)

    step_idx = init_step_idx + 1
    _step(hparam, UE_list, BS_list, shadowFad_dB_map, step_idx)