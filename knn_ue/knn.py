def compute_intr_reward(self, obs, action, next_obs, step):
    rep = self.icm.get_rep(obs, action)
    reward = self.pbe(rep)
    reward = reward.reshape(-1, 1)
    return reward


# particle-based entropy
rms = utils.RMS(self.device)
self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                     self.device)

