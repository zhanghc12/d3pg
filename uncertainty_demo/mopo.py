import numpy as np
from mbpo_py.model_replicate_offline import EnsembleDynamicsModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(dirname, version=269):
    env_model = EnsembleDynamicsModel(7, 5, 1, 1, 1,
                                      200,
                                      use_decay=True)
    env_model.ensemble_model.to(device)

    env_model.load(dirname + str(version), load_cpu=True)
    return env_model



def get_uncertainty(model, obs, act):
    inputs = np.concatenate((obs, act), axis=-1)
    ensemble_model_means, ensemble_model_vars = model.predict(inputs)

    return np.mean(np.max(ensemble_model_vars, axis=0), axis=-1, keepdims=False) # net * batch_size * (state_dim + 1)

