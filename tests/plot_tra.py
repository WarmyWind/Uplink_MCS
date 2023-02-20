from config.hyperparam import hparams
from core.data_init_utils import load_UE_posi, build_network_topo
import numpy as np
import matplotlib.pyplot as plt

def plot_hexgon(ax, center, dist):
    radius = dist/np.sqrt(3)
    for _center in center:
        point_list=[]
        for angle in np.arange(0, 2*np.pi, np.pi/3):
            point_list.append(_center + radius*np.exp(1j*angle))

        for i in range(len(point_list)):
            _point1 = point_list[i]
            _point2 = point_list[(i+1) % len(point_list)]
            ax.plot(np.real([_point1, _point2]), np.imag([_point1, _point2]), color='silver')

    return ax

def plot_UE_trajectory(Macro_Posi, UE_tra, label_list=None, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.scatter(np.real(Macro_Posi), np.imag(Macro_Posi), label='Macro BS')
    dist = np.abs(Macro_Posi[0]-Macro_Posi[1])
    ax = plot_hexgon(ax, Macro_Posi, dist)
    if len(UE_tra.shape) == 2:
        for i in range(UE_tra.shape[-1]):
            _UE_tra = UE_tra[:, i]
            _UE_tra = _UE_tra[np.where(_UE_tra != None)]
            if label_list == None:
                ax.plot(np.real(_UE_tra.tolist()), np.imag(_UE_tra.tolist()), label='User{}'.format(i))
            else:
                ax.plot(np.real(_UE_tra.tolist()), np.imag(_UE_tra.tolist()), label=label_list[i])
    elif len(UE_tra.shape) == 1:
        UE_tra = UE_tra[np.where(UE_tra != None)]
        ax.plot(np.real(UE_tra.tolist()), np.imag(UE_tra.tolist()), label='User')

    plt.legend()
    # plt.show()
    return ax


if __name__ == '__main__':
    UE_tra = load_UE_posi(hparams)
    UE_tra = np.reshape(UE_tra, (UE_tra.shape[2],-1))
    BS_posi = build_network_topo(hparams)
    ax = plot_UE_trajectory(BS_posi, UE_tra)
    plt.show()
