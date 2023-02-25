import numpy as np
from config.settings import *

def milimeter_channel(UE, BS):
    N_t, N_r = 16, 1
    N_c, N_ray = 2, 10
    alpha2, beta2, sigma2 = 61.4, 2, 5.8

    num_frame = 10
    time_slot = 0.001
    R_factor = 0
    f_c = 2.8e10
    speed_light = 3e8
    BS_posi, UE_posi = list(BS.posi), list(UE.posi)
    BS_posi.append(10)
    UE_posi.append(1.5)
    BS_coord = np.array(BS_posi)
    UE_coord = np.array(UE_posi)

    # Calculate doppler
    speed_UE = UE.speed
    max_doppler = speed_UE * f_c / speed_light
    nor_doppler = max_doppler * num_frame

    # NLOS path
    # -----------NLOS AOD------------
    E_aod = np.random.uniform(-np.pi / 2, np.pi / 2, size=N_c)
    sigma_aod = 10 * np.pi / 180
    b = sigma_aod / np.sqrt(2)
    a = np.random.rand(N_c, N_ray) - 0.5
    aod_rad = np.tile(E_aod, (N_ray, 1)).T - b * np.sign(a) * np.log(1 - 2 * np.abs(a))

    signature_t = np.arange(N_t)
    H = np.zeros((num_frame, N_r, N_t), dtype=complex)
    H_ray = np.zeros((num_frame, N_r, N_t, N_c, N_ray), dtype=complex)
    H_cl = np.zeros((num_frame, N_r, N_t, N_c), dtype=complex)
    H_NLOS = np.zeros((num_frame, N_r, N_t), dtype=complex)
    H_LOS = np.zeros((num_frame, N_r, N_t), dtype=complex)

    D_coord = UE_coord - BS_coord
    D_2D = np.sqrt(D_coord[0] ** 2 + D_coord[1] ** 2)
    D_3D = np.sqrt(D_2D ** 2 + (D_coord[2]) ** 2)
    PL = np.sqrt(10 ** (-0.1 * (alpha2 + 10 * beta2 * np.log10(D_3D))))
    complex_gain = (np.random.randn(N_c, N_ray) + 1j * np.random.randn(N_c, N_ray)) / np.sqrt(2)  # 瑞利项

    for t in range(num_frame):
        for i in range(N_c):
            for j in range(N_ray):
                doppler = max_doppler * np.sin(aod_rad[i, j])
                try:
                    H_ray[t, :, :, i, j] = np.exp(1j * 2 * np.pi * doppler * t * time_slot) * complex_gain[i, j] \
                                       * np.exp(np.sin(aod_rad[i, j]) * 1j * np.pi * signature_t) / np.sqrt(N_t * N_r)
                except:
                    pass
    H_cl = np.sum(H_ray, axis=-1)
    H_NLOS[:, :, :] = np.sqrt(N_t * N_r / N_c / N_ray) * np.sum(H_cl, axis=-1)

    # LOS path
    # -----------LOS AOA,AOD------------%
    aod_LOS = np.arctan2(UE_coord[1], UE_coord[0])
    doppler = -max_doppler * np.sin(aod_LOS)
    for t in range(num_frame):
        H_LOS[t, :, :] = np.exp(1j * 2 * np.pi * doppler * time_slot * t) * np.exp(
            np.sin(aod_LOS) * 1j * np.pi * signature_t).T
    H = np.sqrt(R_factor / (1 + R_factor)) * H_LOS + np.sqrt(1 / (1 + R_factor)) * H_NLOS
    return H


def large_scale_fading(hparam, BS_list, UE_list, shadow_map):
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
            large_fading_dB = get_large_fading_dB(hparam, BS_list[iBS], UE_list[iUE], shadow_map)
            try:
                large_scale_fading_dB[iBS, iUE] = large_fading_dB[0]
            except:
                large_scale_fading_dB[iBS, iUE] = large_fading_dB

    # large_scale_channel = 10 ** (-large_scale_fading_dB / 20)
    # print('大尺度衰落(dB)：',large_scale_fading_dB[:,0])
    return large_scale_fading_dB


def get_large_fading_dB(hparam, BS, UE, shadow_map):
    # if not UE.active:mlarge_fading_dB = np.Inf
    large_fading_dB = get_large_fading_dB_from_posi(hparam, UE.posi, BS.posi, BS.no, shadow_map)
    return large_fading_dB


def get_large_fading_dB_from_posi(hparam, UE_posi, BS_posi, BS_no, shadow_map):
    antGain = Macro.antGaindB
    dFactor = Macro.dFactordB
    pLoss1m = Macro.pLoss1mdB

    distServer = np.abs(UE_posi - BS_posi)  # 用户-基站距离

    if shadow_map is None:
        if hparam.shadow_type == 'none':
            _shadow = 0
        elif hparam.shadow_type == 'random':
            try:
                size = len(UE_posi)
            except:
                size = 1
            _shadow = np.random.normal(0, hparam.shadow_std, size=size)
        else:
            raise Exception('Unsupported shadow type')
    else:
        origin_x_point = np.real(hparam.origin_point)
        origin_y_point = np.imag(hparam.origin_point)
        x_temp = (np.ceil((np.real(UE_posi) - origin_x_point) / hparam.posi_resol)).astype(np.int) - 1
        y_temp = (np.ceil((np.imag(UE_posi) - origin_y_point) / hparam.posi_resol)).astype(np.int) - 1
        # x_temp = np.min((shadow_map.shape[1] - 2, x_temp))
        # y_temp = np.min((shadow_map.shape[2] - 2, y_temp))
        # x_temp = np.max((0, x_temp))
        # y_temp = np.max((0, y_temp))
        _shadow = shadow_map[BS_no][x_temp, y_temp]
    large_fading_dB = pLoss1m + dFactor * np.log10(distServer) + _shadow - antGain
    return large_fading_dB


def small_scale_fading(nBS, nUE, nRB, nNt, fading_model='Rayleigh'):
    small_H = np.ones((nBS, nUE, nRB, nNt), dtype=np.complex_)

    if fading_model == 'Rayleigh':
        np.random.seed()
        small_H = (np.random.randn(nBS, nUE, nRB, nNt) + 1j*np.random.randn(nBS, nUE, nRB, nNt)) / np.sqrt(2)

    return small_H


if __name__ == '__main__':
    from config.hyperparam import hparams
    from lib.common_class import UE, BS
    np.random.seed(hparams.seed)

    UE1 = UE(no=0, tra=[50 + 1j * 50])
    UE1.posi = [50, 50]
    UE1.speed = 5
    BS1 = BS(no=0, posi=[0, 0])
    H = milimeter_channel(UE1, BS1)
