import numpy as np
from core.get_channel import get_large_fading_dB_from_posi
from config.settings import *


def _one_cell_with_two_road(dist, road_width, UE_num, velocity, time_interval, duration):
    '''
    In this function, the BS position is supposed as (0, 0),
    and the upper road and lower road are both dist/4 away from the BS.
    Generate the UE trajactory on two roads without getting out of the cell.
    '''
    # Assert UEs won't get out of the cell
    if velocity * duration > dist*np.sqrt(3)/4:
        duration = np.floor(dist*np.sqrt(3)/4 / velocity)
    # assert velocity * duration <= dist*np.sqrt(3)/4

    # random upper road or lower road
    road_idx = np.random.binomial(1, 0.5, UE_num)* 2 - 1

    # random going left or right
    direction_idx = np.random.binomial(1, 0.5, UE_num) * 2 - 1

    # random y bias on road
    y_bias = np.random.uniform(0, 1, UE_num) * road_width

    # random x bias when start
    x_bias = np.random.uniform(0, 1, UE_num) * dist*np.sqrt(3)/4

    # calculate start point
    start_point = -direction_idx * x_bias + 1j * (road_idx * dist/4 - road_idx*y_bias)

    UE_tra = []
    for idx in range(UE_num):
        _x_transloc = direction_idx[idx] * np.arange(0, duration*velocity, time_interval*velocity)
        _UE_tra = start_point[idx] + _x_transloc
        UE_tra.append(_UE_tra)

    return np.array(UE_tra)



def generate_UE_tra(BS_posi_list, dist, road_width, nUE_per_cell, velocity, time_interval, duration):
    '''
    Generate trajectory of UEs
    '''
    UE_tra = []
    for _BS_posi in BS_posi_list:
        _UE_tra = _one_cell_with_two_road(dist, road_width, nUE_per_cell, velocity, time_interval, duration)
        _UE_tra += _BS_posi
        UE_tra.append(_UE_tra)

    return np.array(UE_tra)


def generate_large_scale_fading(hparam, BS_posi, UE_tra, shadow_map):
    large_fading_dB = []
    for BS_idx in range(UE_tra.shape[0]):
        for UE_idx in range(UE_tra.shape[1]):
            _UE_tra = UE_tra[BS_idx, UE_idx, :]
            _large_fading_dB = get_large_fading_dB_from_posi(hparam, _UE_tra, BS_posi[BS_idx], BS_idx, shadow_map)
            large_fading_dB.append(_large_fading_dB)

    return np.array(large_fading_dB)


def generate_SINR(lower, upper, step=np.pi/20, length=5000, noise_std=1):
    '''
    Generate SINR_dB with Guassian noise.
    Use Sine to generate values between [lower, upper] + noise
    '''
    x = np.arange(step, length * step, step)

    ideal_SINR = np.sin(x) * (upper - lower)/2 + (upper+lower)/2
    noisy_SINR = ideal_SINR + np.random.normal(0, noise_std, size=ideal_SINR.shape)

    return noisy_SINR


if __name__ == '__main__':
    def generate_example_UE_tra(save=True):
        one_cell_UE_tra = _one_cell_with_two_road(dist=150, road_width=20, UE_num=50, velocity=15,
                                         time_interval=0.1, duration=5)
        print('one_cell_UE_tra shape:', one_cell_UE_tra.shape)
        from core.network_topo import cell_struction
        BS_posi = cell_struction(7, 150)
        from tests.plot_tra import plot_UE_trajectory
        ax = plot_UE_trajectory(BS_posi, (one_cell_UE_tra+BS_posi[6]).T)
        import matplotlib.pyplot as plt
        plt.show()
        UE_tra = generate_UE_tra(BS_posi, dist=150, road_width=20, nUE_per_cell=10, velocity=15,
                                         time_interval=0.1, duration=5)
        print('UE_tra shape:', UE_tra.shape)
        if save:
            np.save('../db/example_UE_tra.npy', UE_tra, allow_pickle=True)

    def generate_large_fading_dataset(save=True):
        from config.hyperparam import hparams
        from core.data_init_utils import load_shadowfad_map
        from core.network_topo import cell_struction
        shadowFad_dB_map = load_shadowfad_map(hparams)
        BS_posi = cell_struction(7, 150)
        UE_tra = generate_UE_tra(BS_posi, dist=150, road_width=20, nUE_per_cell=100, velocity=15,
                                         time_interval=0.1, duration=5)
        large_fading_dB_data = generate_large_scale_fading(hparams, BS_posi, UE_tra, shadowFad_dB_map)
        if save:
            np.save('../db/large_fading_dB_data.npy', large_fading_dB_data, allow_pickle=True)

    generate_large_fading_dataset(save=True)
