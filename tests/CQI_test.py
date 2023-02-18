from core.CSI_cal import *
from core.CSI_est import estimate_best_CQI
from lib.data_generator import generate_SINR
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

if __name__ == '__main__':
    # SINR_dB_threshold = [-9.478,-6.658,-4.098,-1.798, 0.399,
    #                       2.424, 4.489, 6.367, 8.456,10.266,
    #                      12.218,14.122,15.849,17.786,19.809]

    step = np.pi/5
    SINR_dB_arr = generate_SINR(-1.798, 14.122, step=step, length=5000, noise_std=1)
    plt.plot(np.arange(step, 5000 * step, step), SINR_dB_arr)
    plt.xlim([0,50])
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('SINR(dB)')
    plt.show()

    pred_std_list = [1,2,3,4,5,6,7,8,9,10]  # 预测与真值之间的噪声标准差
    ideal_list = []
    nonpred_list = []
    pred_list = []
    best_list = []
    for pred_std in pred_std_list:
        pred_SINR_arr = SINR_dB_arr + np.random.normal(0, pred_std, size=SINR_dB_arr.shape)
        nonpred_SINR_arr = np.concatenate([[SINR_dB_arr[0]],SINR_dB_arr[:-1]])

        ideal_CQI = np.array(list(map(SINR_to_CQI, SINR_dB_arr)))
        pred_CQI = np.array(list(map(SINR_to_CQI, pred_SINR_arr)))
        nonpred_CQI = np.array(list(map(SINR_to_CQI, nonpred_SINR_arr)))

        ideal_efficiency = list(map(cal_spectral_efficiency, zip(ideal_CQI, ideal_CQI)))
        pred_efficiency = list(map(cal_spectral_efficiency, zip(pred_CQI, ideal_CQI)))
        nonpred_efficiency = list(map(cal_spectral_efficiency, zip(nonpred_CQI, ideal_CQI)))
        # print("Ideal efficiency:{:.4f} \nPred efficiency:{:.4f} \nnonPred efficiency:{:.4f}"
        #       .format(np.mean(ideal_efficiency), np.mean(pred_efficiency), np.mean(nonpred_eefficiency)))

        std_arr = np.array([pred_std for _ in range(len(pred_SINR_arr))])
        bestest_CQI = np.array(list(map(estimate_best_CQI, zip(pred_SINR_arr, std_arr))))
        best_efficiency = list(map(cal_spectral_efficiency, zip(bestest_CQI, ideal_CQI)))
        # print("After est std, Test efficiency:{:.4f}".format(np.mean(best_efficiency)))
        ideal_list.append(np.mean(ideal_efficiency))
        nonpred_list.append(np.mean(nonpred_efficiency))
        pred_list.append(np.mean(pred_efficiency))
        best_list.append(np.mean(best_efficiency))

    print('Ideal:{:.4f}'.format(ideal_list[0]))
    print('Obsolete:{:.4f}'.format(nonpred_list[0]))
    print('pred:{}'.format(pred_list))
    print('best:{}'.format(best_list))