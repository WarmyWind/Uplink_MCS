from config.settings import SINR_dB_threshold, SpectralEfficiency_list
# from core.CQI_cal import SINR_to_CQI
import numpy as np
from scipy import stats

def cal_expected_efficiency(over_threshold_likelihood, CQI):
    return SpectralEfficiency_list[CQI] * over_threshold_likelihood[CQI]


def estimate_best_CQI(*args):
    if len(args) == 1:
        assert isinstance(args[0], tuple) and len(args[0]) == 2
        SINR, std = args[0]
    elif len(args) == 2:
        SINR, std = args[0], args[1]
    else:
        raise Exception('Invalid args')

    norm = stats.norm(loc=SINR, scale=std)
    threshold_cdf = norm.cdf(SINR_dB_threshold)
    over_threshold_likelihood = 1 - threshold_cdf
    over_threshold_likelihood = np.concatenate([[1],over_threshold_likelihood])
    # piecewise_likelihood = threshold_cdf[1:]-threshold_cdf[:-1]

    best_CQI = 0
    best_efficiency = 0
    for _CQI in range(1, 16):
        _efficiency = cal_expected_efficiency(over_threshold_likelihood, _CQI)
        if _efficiency > best_efficiency:
            best_efficiency = _efficiency
            best_CQI = _CQI

    return best_CQI


