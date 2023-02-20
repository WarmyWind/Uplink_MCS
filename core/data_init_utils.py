import numpy as np
from lib.utils import get_data_from_mat
from core.network_topo import cell_struction
from lib.common_class import UE, BS
from config.settings import *

def load_UE_posi(hparam):
    try:
        UE_tra = np.load(hparam.UE_tra_filepath, allow_pickle=True)
    except:
        UE_tra = np.load('../'+hparam.UE_tra_filepath, allow_pickle=True)
    UE_tra = UE_tra[:,:5,:]
    return UE_tra


def build_network_topo(hparam):
    if hparam.network_topo == 'cell':
        return cell_struction(hparam.nBS, hparam.dist)

    else:
        raise Exception('Unsupported network toopology')


def load_shadowfad_map(hparam):
    if hparam.shadow_type == 'none' or hparam.shadow_type == 'random':
        return None
    elif hparam.shadow_type == 'fixed':
        if hparam.shadow_filepath[-3:] == 'mat':
            try:
                shadowFad_dB = get_data_from_mat(hparam.shadow_filepath, 'shadowFad_dB')
            except:
                shadowFad_dB = get_data_from_mat('../'+hparam.shadow_filepath, 'shadowFad_dB')
        elif hparam.shadow_filepath[-3:] == 'npy':
            try:
                shadowFad_dB = np.load(hparam.shadow_filepath, allow_pickle=True)
            except:
                shadowFad_dB = np.load('../'+hparam.shadow_filepath, allow_pickle=True)
        else:
            raise Exception('Unsupported file')
        return shadowFad_dB
    else:
        raise Exception('Unsupported shadow type')


def load_NN_model(hparam):
    # TODO: Load NN model for prediction
    return None


def init_UE(hparam):
    UE_tra = load_UE_posi(hparam)
    UE_list = []
    if hparam.HO_type == 'none':
        # No handover, one UE staticly connects to one BS
        assert len(UE_tra.shape) == 3
        for BS_idx in range(UE_tra.shape[0]):
            for UE_idx in range(UE_tra.shape[1]):
                _UE_no = BS_idx*UE_tra.shape[1]+UE_idx
                _UE = UE(no=_UE_no, tra=UE_tra[BS_idx, UE_idx, :])
                _UE.serv_BS = BS_idx
                _UE.HO_type = 'none'
                UE_list.append(_UE)

    elif hparam.HO_type == 'ideal':
        # TODO
        pass
    else:
        raise Exception('Invalid HO type!')

    return UE_list


def init_BS(hparam):
    BS_posi = build_network_topo(hparam)
    BS_list = []
    for _BS_no, _BS_posi in enumerate(BS_posi):
        _BS = BS(no=_BS_no, posi=BS_posi[_BS_no])
        BS_list.append(_BS)

    return BS_list


def init_all(hparam):
    '''
    Initialize UE, BS, shadow fading and NN model(if use)
    :param hparam:
    :return:
    '''

    UE_list = init_UE(hparam)
    BS_list = init_BS(hparam)
    shadowFad_dB_map = load_shadowfad_map(hparam)

    return UE_list, BS_list, shadowFad_dB_map

