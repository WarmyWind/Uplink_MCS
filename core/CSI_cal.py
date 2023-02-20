import numpy as np
from config.settings import *

def calculate_SINR_dB(receive_power, interference_power, noise):
    SINR = receive_power/(interference_power+noise)
    if SINR == 0:
        return -np.Inf
    return 10*np.log10(SINR)

def SINR_to_CQI(SINR_dB):
    '''
    Map SINR_dB to CQI index
    :param SINR_dB:
    :return:
    '''

    if SINR_dB >= 19.809:
        return 15
    else:
        return np.where(SINR_dB < np.array(SINR_dB_threshold))[0][0]

def CQI_to_SpectralEfficiency(CQI):
    '''
    Map CQI index to spectral efficiency
    :param CQI:
    :return:
    '''
    assert 0 <= CQI <= 15

    return SpectralEfficiency_list[CQI]

def cal_spectral_efficiency(*args):
    if len(args) == 1:
        assert isinstance(args[0], tuple) and len(args[0]) == 2
        CQI_pred, CQI_real = args[0]
    elif len(args) == 2:
        CQI_pred, CQI_real = args[0], args[1]
    else:
        raise Exception('Invalid args')

    if CQI_pred <= CQI_real:
        return CQI_to_SpectralEfficiency(CQI_pred)
    else:
        return 0


def get_spec_effi(UE_list):
    ideal_spec_effi_list = []
    spec_effi_list = []
    for _UE in UE_list:
        _SINR_dB = _UE.SINR_dB
        cal_spectral_efficiency()



if __name__ == '__main__':
    # x = np.array([1,3,5])
    # CQI = np.array(list(map(SINR_to_CQI, x)))
    # print(CQI)
    #
    # x = (1,2)
    # c = cal_spectral_efficiency(x)
    # print(c)
    #
    # a = [2,4,6]
    # b = [2,3,5]
    #
    # SE = np.array(list(map(cal_spectral_efficiency, zip(a, b))))
    # print(SE)

    print(cal_spectral_efficiency(3,4))

