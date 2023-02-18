from core.get_channel import large_scale_channel, small_scale_fading
from core.CSI_cal import calculate_SINR_dB
from lib.utils import power_to_dB
from config.settings import *
import numpy as np

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


def update_CSI(hparam, UE_list, BS_list, shadowFad_dB_map):
    large_scale_fading_dB = large_scale_channel(hparam, BS_list, UE_list, shadowFad_dB_map)  # (BS, UE)
    update_update_arrival_power(UE_list, large_scale_fading_dB)
    update_interference(UE_list, BS_list, large_scale_fading_dB)
    update_SINR(UE_list, BS_list)
    return





def update_update_arrival_power(UE_list, channel_fading_dB):
    # Update arrival power to the serving_BS for every UE
    for _UE in UE_list:
        _UE.history_arrival_power.append(_UE.arrival_power)
        _UE.arrival_power = _UE.Ptmax_dBm + channel_fading_dB[_UE.serv_BS, _UE.no]

    return


def update_interference(UE_list, BS_list, channel_fading_dB):
    for _BS in BS_list:
        _serv_UE_list = _BS.serv_UE_list
        UE_no_arr = np.arange(len(UE_list))
        itf_source = np.delete(UE_no_arr, _serv_UE_list)
        _BS.interference = 0
        for _UE_no in itf_source:
            _BS.history_interference.append(_BS.interference)
            _BS.interference += UE_list[_UE_no].Ptmax_dBm + channel_fading_dB[_BS.no, _UE_no]
    return


def update_SINR(UE_list, BS_list):
    for _UE in UE_list:
        _BS = BS_list[UE_list.serv_BS]
        _UE.SINR_dB = calculate_SINR_dB(power_to_dB(_UE.arrival_power), power_to_dB(_BS.interference), sigma2)

    return


