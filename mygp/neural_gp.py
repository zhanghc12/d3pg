import torch
import gpytorch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam


class FeatureExtractor(nn.Module):
    """
        Feature preprocess
    """
    def __init__(self, state_dim, action_dim, num_feature):
        super(FeatureExtractor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, num_feature)

    def forward(self, data):
        q = F.relu(self.l1(data))
        q = F.relu(self.l2(q))
        #return spectral_norm(F.relu(self.l3(q)), norm_bound=0.95, n_power_iterations=1) # todo: if relu or not
        return self.l3(q)


class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    """
        Multi-output gp, without neural network
    """
    def __init__(self, inducing_points, num_tasks=4, enable_ngd=True):
        # Let's use a different set of inducing points for each task
        # inducing_points = torch.rand(num_tasks, 16, 1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        if enable_ngd:
            variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
            )
            # variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        else:
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
            )

        # meanfield? or ngd meanfield to define the other one/
        # variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(-2))

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NeuralGP(gpytorch.Module):
    def __init__(self, feature_extractor, inducing_points, num_tasks, enable_ngd):
        super(NeuralGP, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = IndependentMultitaskGPModel(inducing_points=inducing_points, num_tasks=num_tasks, enable_ngd=enable_ngd)

        # self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        # self.grid_bounds = grid_bounds
        # self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        # features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        # features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

'''
# define the model and likelihood
train_x = None
train_y = None
state_dim = 1
action_dim = 1
device = 'cuda'

enable_ngd = True
feature_extractor = FeatureExtractor(state_dim=state_dim, action_dim=action_dim, num_feature=128)
inducing_points = train_x[:5000, :] # todo
inducing_points = feature_extractor(inducing_points)  # initialize the inducing points
inducing_points = inducing_points.unsqueeze(0).repeat(train_y.shape[1], 1, 1)  # y_shape * ibatch * 128

model = NeuralGP(feature_extractor, inducing_points, num_tasks=train_y.shape[1], enable_ngd=enable_ngd)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1]).to(device)

variational_ngd_optimizer = gpytorch.optim.NGD(model.gp_layer.variational_parameters(), num_data=train_y.size(0), lr=0.1)

lr = 0.1
if enable_ngd:
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
else:
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)

mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=train_y.size(0)).to(device)


# todo 3: in the loop, fix the other things
'''