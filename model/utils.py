from model.bnn.model import BayesianRegressor
from model.mcdropout.model import MCDropoutRegressor, log_gaussian_loss
from model.ensemble.model import BootstrapEnsemble
from model.edl.model import EvidentialRegressor
from model.Dataset.load_dataset import *
from lib.utils import get_device

def get_dataloader(hparams, norm=True):
    if hparams.task == 'large_channel_predict':
        dl_train, dl_valid, dl_test = load_large_channel(hparams.dataset_path, hparams.input_dim, hparams.output_dim,
                                                         test_size=0.2, valid_size=0.2, train_batch_size=hparams.batch_size,
                                                         valid_batch_size=hparams.batch_size, seed=hparams.seed, norm=norm)
    elif hparams.task == 'uci_wine':
        dl_train, dl_valid, dl_test = load_uci_wine(test_size=0.25, valid_size=0,
                                                    train_batch_size=hparams.batch_size,
                                                    valid_batch_size=hparams.batch_size, seed=hparams.seed)
    elif hparams.task == 'uci_crime':
        dl_train, dl_valid, dl_test = load_uci_crime(test_size=0.25, valid_size=0,
                                                     train_batch_size=hparams.batch_size,
                                                     valid_batch_size=hparams.batch_size, seed=hparams.seed)
    elif hparams.task == 'uci_energy':
        try:
            gap = hparams.data_gap
        except:
            gap = 1
        dl_train, dl_valid, dl_test = load_uci_energy(hparams.dataset_path, hparams.input_dim, hparams.output_dim,
                                                         test_size=0.2, valid_size=0.2,
                                                         train_batch_size=hparams.batch_size,
                                                         valid_batch_size=hparams.batch_size,
                                                         seed=hparams.seed, gap=gap)
    else:
        raise Exception('Invalid task!')

    return dl_train, dl_valid, dl_test


def get_abnormal_dataloader(hparams, noisy_feature):
    if hparams.task == 'large_channel_predict':
        pass
    elif hparams.task == 'uci_wine':
        pass
    elif hparams.task == 'uci_crime':
        pass
    elif hparams.task == 'uci_energy':
        try:
            gap = hparams.data_gap
        except:
            gap = 1
        dl_train, dl_valid, dl_test = load_uci_energy(hparams.dataset_path, hparams.input_dim, hparams.output_dim,
                                                         test_size=0.2, valid_size=0.2,
                                                         train_batch_size=hparams.batch_size,
                                                         valid_batch_size=hparams.batch_size,
                                                         seed=hparams.seed, gap=gap, noisy_feature=noisy_feature)
    else:
        raise Exception('Invalid task!')

    return dl_train, dl_valid, dl_test


def build_model(hparams):
    method = hparams.method
    lr = hparams.learning_rate
    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method =='BNN':

        if hparams.noise_estimation == 'none':
            criterion = torch.nn.MSELoss()
            model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, criterion=criterion, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            criterion = log_gaussian_loss
            model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, criterion=criterion, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))


    elif method == 'MCDropout':
        if hparams.noise_estimation == 'none':
            model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                       pdrop=dropout_rate, hidden_depth=hparams.hidden_depth, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                       pdrop=dropout_rate, hidden_depth=hparams.hidden_depth, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))

    elif method == 'Ensemble':
        num_net = hparams.num_net
        if hparams.noise_estimation == 'none':
            model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                      num_net=num_net, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                      num_net=num_net, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))
    elif method == 'EDL':
        model = EvidentialRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth)
    else:
        raise Exception('Invalid method!')

    return model


def load_model(root_path, norm_mean, norm_std):
    class Dotdict(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

    import json
    root_path = root_path
    with open(root_path + '/hparams.json') as file:
        hparams_dict = json.load(file)
    hparams_dict["output_dir"] = root_path
    # hparams_dict["norm"] = False
    # hparams_dict["sample_nbr"] = 5
    hparams = Dotdict(hparams_dict)

    model = build_model(hparams)
    method = hparams.method
    device = get_device()
    if method != "Ensemble":
        state_dict = torch.load(hparams.output_dir+'/model.pt')
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model.load(hparams.output_dir)
        model.to(device)
    model.norm_mean = norm_mean
    model.norm_std = norm_std
    return model

def uncertainty_model_predict(inputs, model):
    norm_mean, norm_std = model.norm_mean, model.norm_std
    inputs = (np.array(inputs) - norm_mean) / norm_std
    inputs = np.array([inputs])
    import torch
    x = torch.tensor(inputs).float().to(get_device())
    pred, std = model.predict(x)
    pred = pred * norm_std + norm_mean
    std = std * norm_std
    try:
        return np.array(pred.detach().cpu()), np.array(std.detach().cpu())
    except:
        return np.array(pred), np.array(std)
