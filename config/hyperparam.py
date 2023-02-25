import argparse
from datetime import datetime
from lib.utils import get_results_directory, Hyperparameters, set_seed
import numpy as np
parser = argparse.ArgumentParser()

########################### 通用参数 #############################
# parser.add_argument(
#     "--method",
#     default="CNN",
#     choices=["BNN", "MCDropout", "Ensemble", "EDL", 'DNN', 'CNN'],
#     type=str,
#     help="Specify method",
# )
# parser.add_argument(
#     "--task",
#     default='mnist',
#     choices=["HO_predict", "cifar10", "mnist"],
#     type=str,
#     help="Specify task"
# )
parser.add_argument("--step_end_idx", type=int, default=40, help="Number of simulation steps")
parser.add_argument("--power_est_method",
                    type=str,
                    default='NN',
                    choices=["ideal", "outdated", "NN"],
                    help="Method of arrival power estimation",
                    )
parser.add_argument("--itf_est_method",
                    type=str,
                    default='ideal',
                    choices=["ideal", "outdated", "NN"],
                    help="Method of interference estimation",
                    )
parser.add_argument("--CQI_est_method",
                    type=str,
                    default='best',
                    choices=["direct", "best"],
                    help="Method of CQI estimation",
                    )
parser.add_argument("--nBS", type=int, default=7, help="Number of BS")
parser.add_argument("--network_topo", type=str, default='cell')
parser.add_argument("--dist", type=int, default=150, help="Distance between two BSs")
parser.add_argument("--shadow_type",
                    type=str,
                    default='fixed',
                    choices=['none', 'random', 'fixed'],
                    )
parser.add_argument("--shadow_std", type=float, default=8)

parser.add_argument(
    "--shadow_filepath",
    default="db/std8_corr50.mat",
    help="Shadowfad file path",
)
parser.add_argument("--posi_resol", type=int, default=1.5, help="Position resolution in shadow fading map")
parser.add_argument("--origin_point", type=complex, default=-2.5*150/np.sqrt(3)-1j*225, help="Origin point in shadow fading map")

parser.add_argument(
    "--UE_tra_filepath",
    default="db/example_UE_tra.npy",
    help="UE trajectory file path",
)
parser.add_argument("--HO_type", type=str, default='none', choices=["none", "ideal"], help="HO type")

timestamp = datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")
# parser.add_argument(
#     "--output_dir",
#     default='runs/' + parser.get_default("task") + '/' + parser.get_default("method") + '/' + timestamp,
#     type=str,
#     help="Specify output directory"
# )

parser.add_argument("--seed", default=42, type=int, help="Seed to use for training")

########################### BNN\MCDropout参数 ##################################
parser.add_argument("--sample_nbr", type=int, default=5, help="Number of MC samples to get loss")

########################### MCDropout参数 #############################
parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

########################### Ensemble参数 ###############################
parser.add_argument("--num_net", type=int, default=10, help="Total number of nets of ensemble")
parser.add_argument("--dataset_ratio", type=float, default=0.5, help="Dataset ratio for bagging")

########################### EDL参数 ###############################
parser.add_argument("--edl_loss", type=str, default='digamma', help="Loss type of EDL, digamma or log or mse")
parser.add_argument("--annealing_step", type=int, default=200, help="Annealing step of EDL")

args = parser.parse_args()
hparams = Hyperparameters(**vars(args))
