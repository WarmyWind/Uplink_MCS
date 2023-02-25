from core.get_channel import large_scale_fading, small_scale_fading
from core.CSI_cal import calculate_SINR_dB, SINR_to_CQI, cal_spectral_efficiency
from core.CSI_est import estimate_best_CQI
from lib.utils import dBm_to_power, power_to_dBm
from config.settings import *
from model.utils import uncertainty_model_predict

import numpy as np


def update_CSI(hparam, UE_list, BS_list, shadowFad_dB_map):
    large_scale_fading_dB = large_scale_fading(hparam, BS_list, UE_list, shadowFad_dB_map)  # (BS, UE)

    update_arrival_power(UE_list, large_scale_fading_dB)
    update_est_arrival_power(hparam, UE_list)

    update_interference(UE_list, BS_list, large_scale_fading_dB)
    update_est_interference(hparam, BS_list)

    update_SINR(UE_list, BS_list)
    update_est_SINR(UE_list, BS_list)
    return


def update_CQI(hparam, UE_list):
    for _UE in UE_list:
        _UE.ideal_CQI = SINR_to_CQI(_UE.SINR_dB)
        if hparam.CQI_est_method == 'direct':
            _UE.est_CQI = SINR_to_CQI(_UE.est_SINR_dB)
        elif hparam.CQI_est_method == 'best':
            if _UE.est_SINR_dB_std == 0:
                _UE.est_CQI = SINR_to_CQI(_UE.est_SINR_dB)
            else:
                _UE.est_CQI = estimate_best_CQI(_UE.est_SINR_dB, _UE.est_SINR_dB_std)
        else:
            raise Exception("Invalid CQI estimation method!")

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


def update_arrival_power(UE_list, channel_fading_dB):
    # Update arrival power to the serving_BS for every UE
    for _UE in UE_list:
        if _UE.arrival_power_dBm != -np.Inf:
            _UE.history_arrival_power_dBm.append(_UE.arrival_power_dBm)
        _UE.arrival_power_dBm = _UE.Ptmax_dBm - channel_fading_dB[_UE.serv_BS, _UE.no]
    return


def update_interference(UE_list, BS_list, channel_fading_dB):
    for _BS in BS_list:
        if _BS.interference != np.Inf:
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
        _UE.SINR_dB = calculate_SINR_dB(dBm_to_power(_UE.arrival_power_dBm),
                                        _BS.interference, sigma2)
    return


def update_est_arrival_power(hparam, UE_list):
    if hparam.power_est_method == 'ideal':
        for _UE in UE_list:
            _UE.est_arrival_power_dBm = _UE.arrival_power_dBm
            _UE.est_arrival_power_dBm_std = 0
    elif hparam.power_est_method == 'outdated':
        for _UE in UE_list:
            try:
                _UE.est_arrival_power_dBm = _UE.history_arrival_power_dBm[-1]
            except:
                _UE.est_arrival_power_dBm = _UE.arrival_power_dBm
            _UE.est_arrival_power_dBm_std = 0
    elif hparam.power_est_method == 'NN':
        for _UE in UE_list:
            history_arrival_power_dBm = _UE.history_arrival_power_dBm[-hparam.large_fading_NN.input_dim:]
            if len(history_arrival_power_dBm) < hparam.large_fading_NN.input_dim:
                # If history record is not enough for NN, use outdated infomation
                try:
                    _UE.est_arrival_power_dBm = _UE.history_arrival_power_dBm[-1]
                except:
                    _UE.est_arrival_power_dBm = _UE.arrival_power_dBm
                _UE.est_arrival_power_dBm_std = 0
            else:
                history_fading_dB = _UE.Ptmax_dBm - np.array(history_arrival_power_dBm)
                pred_fading_dB, pred_std = uncertainty_model_predict(history_fading_dB, hparam.large_fading_NN)
                _UE.est_arrival_power_dBm = _UE.Ptmax_dBm - pred_fading_dB.squeeze()
                _UE.est_arrival_power_dBm_std = pred_std.squeeze()
        pass
    else:
        raise Exception('Unsupported estimation method!')
    return


def update_est_interference(hparam, BS_list):
    if hparam.itf_est_method == 'ideal':
        for _BS in BS_list:
            _BS.est_interference = _BS.interference
            _BS.est_interference_std = 0
    elif hparam.itf_est_method == 'outdated':
        for _BS in BS_list:
            _BS.est_interference = _BS.history_interference[-1]
            _BS.est_interference_std = 0
    elif hparam.itf_est_method == 'NN':
        # TODO
        pass
    else:
        raise Exception('Unsupported estimation method!')
    return


def update_est_SINR(UE_list, BS_list):
    for _UE in UE_list:
        _BS = BS_list[_UE.serv_BS]
        _UE.est_SINR_dB = calculate_SINR_dB(dBm_to_power(_UE.est_arrival_power_dBm),
                                            _BS.est_interference, sigma2)
        _UE.est_SINR_dB_std = _UE.est_arrival_power_dBm_std
    return

