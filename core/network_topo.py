import numpy as np

def cell_struction(nCell, Dist):
    if nCell == 7:
        x_interval = Dist/2*np.sqrt(3)
        return np.array([0+1j*Dist, -x_interval+1j*Dist/2, x_interval+1j*Dist/2,
                         0+1j*0,
                         -x_interval-1j*Dist/2, x_interval-1j*Dist/2, 0-1j*Dist])
    else:
        raise Exception('Unsupported nCell!')