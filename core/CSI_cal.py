import numpy as np
from config.settings import *

def calculate_SINR_dB(receive_power, interference_power, noise):
    SINR = receive_power/(interference_power+noise)
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
    assert len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2
    CQI_pred, CQI_real = args[0]
    if CQI_pred <= CQI_real:
        return CQI_to_SpectralEfficiency(CQI_pred)
    else:
        return 0


if __name__ == '__main__':
    x = np.array([1,3,5])
    CQI = np.array(list(map(SINR_to_CQI, x)))
    print(CQI)

    x = (1,2)
    c = cal_spectral_efficiency(x)
    print(c)

    a = [2,4,6]
    b = [2,3,5]

    SE = np.array(list(map(cal_spectral_efficiency, zip(a, b))))
    print(SE)