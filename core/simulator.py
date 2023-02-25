from core.info_update import update_posi, update_access, update_CSI, update_CQI, update_spec_effi, init_access
from lib.utils import progress_bar
from core.evaluate import eval_spec_effi
import numpy as np

def step(hparam, UE_list, BS_list, shadowFad_dB_map, step_idx):
    # Update real and estimated infomation
    update_posi(UE_list, step_idx)
    update_access(hparam, UE_list, BS_list)
    update_CSI(hparam, UE_list, BS_list, shadowFad_dB_map)
    update_CQI(hparam, UE_list)
    update_spec_effi(UE_list)

    return


def simulate(hparam, UE_list, BS_list, shadowFad_dB_map, init_step_idx):
    init_access(hparam, UE_list, BS_list, shadowFad_dB_map, init_step_idx)

    # step_idx = init_step_idx + 1
    ideal_spec_effi_record, spec_effi_record = [], []
    for step_idx in range(init_step_idx + 1, hparam.step_end_idx):
        # Step
        step(hparam, UE_list, BS_list, shadowFad_dB_map, step_idx)

        # Evaluate
        ideal_spec_effi, spec_effi = eval_spec_effi(UE_list)
        ideal_spec_effi_record.append(ideal_spec_effi)
        spec_effi_record.append(spec_effi)

        progress_bar(step_idx / (hparam.step_end_idx - 1) * 100)

    ideal_spec_effi_record = np.array(ideal_spec_effi_record)
    spec_effi_record = np.array(spec_effi_record)
    print('Ideal spec effi:{}\n Eval spec effi:{}'
          .format(np.mean(ideal_spec_effi_record[5:,:]), np.mean(spec_effi_record[5:,:])))