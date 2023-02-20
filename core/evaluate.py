import numpy as np

def eval_spec_effi(UE_list):
    ideal_spec_effi = []
    spec_effi = []
    for _UE in UE_list:
        ideal_spec_effi.append(_UE.ideal_spec_effi)
        spec_effi.append(_UE.spec_effi)

    return ideal_spec_effi, spec_effi
