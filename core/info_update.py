from core.get_channel import large_scale_fading, small_scale_fading
from core.CSI_cal import calculate_SINR_dB, SINR_to_CQI, cal_spectral_efficiency
from lib.utils import dBm_to_power
from config.settings import *

import numpy as np


def update_CSI(hparam, UE_list, BS_list, shadowFad_dB_map):
    large_scale_fading_dB = large_scale_fading(hparam, BS_list, UE_list, shadowFad_dB_map)  # (BS, UE)
    update_update_arrival_power(UE_list, large_scale_fading_dB)
    update_interference(UE_list, BS_list, large_scale_fading_dB)
    update_SINR(UE_list, BS_list)
    update_est_SINR(hparam, UE_list, BS_list)
    return


def update_CQI(UE_list):
    for _UE in UE_list:
        _UE.ideal_CQI = SINR_to_CQI(_UE.SINR_dB)
        _UE.est_CQI = SINR_to_CQI(_UE.est_SINR_dB)
    return


def update_spec_effi(UE_list):
    for _UE in UE_list:
        _UE.ideal_spec_effi = cal_spectral_efficiency(_UE.ideal_CQI, _UE.ideal_CQI)
        _UE.spec_effi = cal_spectral_efficiency(_UE.est_CQI, _UE.ideal_CQI)
    return


def init_access(hparam, UE_list, BS_list, shadowFad_dB_map, step_idx):
    # Initial access: update UEs' position and serving BS and calculate CSI
    for _UE in UE_list:
        _UE.posi = _UE.tra[step_idx]
        if _UE.HO_type == 'none':
            _BS = BS_list[_UE.serv_BS]
            _BS.serv_UE_list.append(_UE.no)
        else:
            raise Exception('Unsupported HO type')

    update_CSI(hparam, UE_list, BS_list, shadowFad_dB_map)
    return


def update_posi(UE_list, step_idx):
    for _UE in UE_list:
        _UE.posi = _UE.tra[step_idx]
    return

def update_access(hparams, UE_list, BS_list):
    if hparams.HO_type == 'none':
        pass
    elif hparams.HO_type == 'ideal':
        # TODO
        pass
    else:
        raise Exception('Invalid HO type')

def update_update_arrival_power(UE_list, channel_fading_dB):
    # Update arrival power to the serving_BS for every UE
    for _UE in UE_list:
        _UE.history_arrival_power.append(_UE.arrival_power)
        _UE.arrival_power = dBm_to_power(_UE.Ptmax_dBm - channel_fading_dB[_UE.serv_BS, _UE.no])
    return


def update_interference(UE_list, BS_list, channel_fading_dB):
    for _BS in BS_list:
        _BS.history_interference.append(_BS.interference)
        _serv_UE_list = _BS.serv_UE_list
        UE_no_arr = np.arange(len(UE_list))
        itf_source = np.delete(UE_no_arr, _serv_UE_list)
        _BS.interference = 0
        for _UE_no in itf_source:
            _BS.interference += dBm_to_power(UE_list[_UE_no].Ptmax_dBm - channel_fading_dB[_BS.no, _UE_no])
    return


def update_SINR(UE_list, BS_list):
    for _UE in UE_list:
        _BS = BS_list[_UE.serv_BS]
        _UE.SINR_dB = calculate_SINR_dB(_UE.arrival_power, _BS.interference, sigma2)
    return


def update_est_SINR(hparam, UE_list, BS_list):
    if hparam.est_method == 'ideal':
        for _UE in UE_list:
            _BS = BS_list[_UE.serv_BS]
            _UE.est_SINR_dB = calculate_SINR_dB(_UE.arrival_power,
                                                _BS.interference, sigma2)
            _UE.est_SINR_dB_std = 0
    elif hparam.est_method == 'outdated':
        for _UE in UE_list:
            _BS = BS_list[_UE.serv_BS]
            _UE.est_SINR_dB = calculate_SINR_dB(_UE.history_arrival_power[-1],
                                                _BS.interference, sigma2)
            _UE.est_SINR_dB_std = 0
    elif hparam.est_method == 'NN':
        # TODO
        pass
    else:
        raise Exception('Unsupported estimation method!')
    return

