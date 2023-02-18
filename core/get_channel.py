import numpy as np
from config.settings import *

def large_scale_channel(hparam, BS_list, UE_list, shadow_map):
    '''
    大尺度信道衰落=路径衰落+阴影衰落
    :param hparam: 仿真参数
    :param BS_list: BS列表，元素是BS类
    :param UE_list: UE列表，元素是UE类
    :param shadow_map: 阴影衰落地图，与地理位置有关
    :return: large_scale_fading_dB，1个二维数组表示BS-UE的大尺度衰落,单位dB
    '''

    nBS = len(BS_list)  # 所有基站数
    nUE = len(UE_list)  # 所有用户数
    large_scale_fading_dB = np.zeros((nBS, nUE))
    for iUE in range(nUE):
        for iBS in range(nBS):
            large_fading_dB = get_large_fading_dB(hparam, BS_list[iBS], UE_list[iUE], shadow_map, hparam.scene)
            large_scale_fading_dB[iBS, iUE] = large_fading_dB

    # large_scale_channel = 10 ** (-large_scale_fading_dB / 20)
    # print('大尺度衰落(dB)：',large_scale_fading_dB[:,0])
    return large_scale_fading_dB


def get_large_fading_dB(hparam, BS, UE, shadow_map):
    # if not UE.active:mlarge_fading_dB = np.Inf
    large_fading_dB = get_large_fading_dB_from_posi(hparam, UE.posi, BS.posi, BS.no, shadow_map, BS.type)
    return large_fading_dB


def get_large_fading_dB_from_posi(hparam, UE_posi, BS_posi, BS_no, shadow_map, BS_type, scene):
    antGain = Macro.antGaindB
    dFactor = Macro.dFactordB
    pLoss1m = Macro.pLoss1mdB

    distServer = np.abs(UE_posi - BS_posi)  # 用户-基站距离

    origin_x_point = hparam.origin_x
    origin_y_point = hparam.origin_y
    x_temp = int(np.ceil((np.real(UE_posi) - origin_x_point) / hparam.posi_resolution)) - 1
    y_temp = int(np.ceil((np.imag(UE_posi) - origin_y_point) / hparam.posi_resolution)) - 1
    x_temp = np.min((shadow_map.map.shape[2] - 2, x_temp))
    y_temp = np.min((shadow_map.map.shape[1] - 2, y_temp))
    x_temp = np.max((0, x_temp))
    y_temp = np.max((0, y_temp))

    # _shadow = shadow_map.map[BS_no][y_temp, x_temp]
    _shadow = shadow_map.map[BS_no][x_temp, y_temp]
    large_fading_dB = pLoss1m + dFactor * np.log10(distServer) + _shadow - antGain
    return large_fading_dB


def small_scale_fading(nBS, nUE, nRB, nNt, fading_model='Rayleigh'):
    small_H = np.ones((nBS, nUE, nRB, nNt), dtype=np.complex_)

    if fading_model == 'Rayleigh':
        np.random.seed()
        small_H = (np.random.randn(nBS, nUE, nRB, nNt) + 1j*np.random.randn(nBS, nUE, nRB, nNt)) / np.sqrt(2)

    return small_H