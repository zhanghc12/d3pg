import torch
import torch.nn as nn

next_state = None
action = None
state_dim = 1
action_dim = 1
reverse_actor_criterion = nn.MSELoss()
reverse_actor = RerverseVaeActor(
    obs_dim=state_dim,
    action_dim=action_dim,
    latent_dim=action_dim * 2
)
reverse_actor_optimizer = torch.optim.Adam(reverse_actor.parameters(), lr=3e-4)

if torch.cuda.is_available():
    recon, mean, std = reverse_actor(next_state, action)
    recon_loss = reverse_actor_criterion(recon, action)
    kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    reverse_actor_loss = recon_loss + 0.5 * kl_loss

    reverse_actor_optimizer.zero_grad()
    reverse_actor_loss.backward()
    reverse_actor_optimizer.step()
