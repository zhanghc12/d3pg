import torch
from mc_dropout.drop import FlattenDropout_Mlp, VAEPolicy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

def build_network(obs_dim, action_dim, dirname, version=9):
    M = 256
    variant = {
        'drop_rate': 0.1,
        'spectral_norm': True
    }

    qf1 = FlattenDropout_Mlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, ],
        drop_rate=variant['drop_rate'],
        spectral_norm=variant['spectral_norm'],
    ).to(device)
    qf2 = FlattenDropout_Mlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, ],
        drop_rate=variant['drop_rate'],
        spectral_norm=variant['spectral_norm'],
    ).to(device)


    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[750, 750],
        latent_dim=action_dim * 2,
    ).to(device)

    qf1_dirname = dirname + '_qf1_' + str(version)
    qf2_dirname = dirname + '_qf2_' + str(version)
    vae_dirname = dirname + '_vae_' + str(version)

    qf1.load_state_dict(torch.load(qf1_dirname, map_location=torch.device('cpu')))
    qf2.load_state_dict(torch.load(qf2_dirname, map_location=torch.device('cpu')))
    vae_policy.load_state_dict(torch.load(vae_dirname, map_location=torch.device('cpu')))
    return qf1, qf2, vae_policy


def get_uncertainty(state, action, qf1, qf2, vae, use_vae=False):
    state_batch = state
    action_batch = action

    state_batch = torch.FloatTensor(state_batch).to(device=device)
    action_batch = torch.FloatTensor(action_batch).to(device=device)

    q_val1, q_val1_var = qf1.multiple(state_batch, action_batch, with_var=True)
    q_val2, q_val2_var = qf2.multiple(state_batch, action_batch, with_var=True)

    q_var = q_val1_var + q_val2_var

    if not use_vae:
        weight = torch.clamp(torch.exp(-0.5 * q_var / 1), 0, 1).squeeze().detach().cpu().numpy()

    # now put vae forward
    # print(weight.shape)

    else:
        sampled_actions, raw_sampled_actions = vae.decode_multiple(state_batch, num_decode=10, device=device)
        distance = (raw_sampled_actions - action_batch.unsqueeze(1).repeat(1, 10, 1)) ** 2
        weight = torch.mean(torch.mean(distance, dim=1), dim=1).detach().cpu().numpy()

    # weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    return weight