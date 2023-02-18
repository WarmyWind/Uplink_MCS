SpectralEfficiency_list = [0, 0.15237, 0.2344, 0.377, 0.6016, 0.877,
                           1.1758, 1.4766, 1.9141, 2.4063, 2.7305,
                           3.3223, 3.9023, 4.5234, 5.1152, 5.5547]

SINR_dB_threshold = [-9.478, -6.658, -4.098, -1.798, 0.399,
                     2.424, 4.489, 6.367, 8.456, 10.266,
                     12.218, 14.122, 15.849, 17.786, 19.809]

nRB = 1
sigma2dBm = -95
sigma2 = 10 ** (sigma2dBm / 10) / 1000

class Macro:
    antGaindB = 0
    dFactordB = 37.6
    pLoss1mdB = 15.3  # 36.8
    shadowdB = 6
    nNt = 16
    Ptmax_dBm = 46
    Ptmax = 10 ** (Ptmax_dBm / 10) / 1000


class Micro:
    antGaindB = 0
    dFactordB = 36.7
    pLoss1mdB = 30.6  # 36.8
    shadowdB = 6


class User:
    nNt = 1
    Ptmax_dBm = 24
    Ptmax = 10 ** (Ptmax_dBm / 10) / 1000

