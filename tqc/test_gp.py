import os
import math

from matplotlib import pyplot as plt
import torch
import gpytorch
from gpytorch import kernels, means, models, mlls, settings
from gpytorch import distributions as distr
import torch.optim as optim


# Training data is 100 points in [0,1] inclusive regularly spaced
train_x1 = torch.linspace(0, 0.4, 50)
train_x2 = torch.linspace(0.6, 1, 50)
train_x = torch.cat([train_x1, train_x2])
train_y = torch.rand(100) * 0.1
# True function is sin(2*pi*x) with Gaussian noise
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

train_x1 = torch.linspace(0.2, 0.8, 50)
train_x2 = torch.linspace(0.2, 0.8, 50)
train_x = torch.cat([train_x1, train_x2])
train_y1 = torch.ones(50) * 0.1
train_y2 = torch.ones(50) * 0.9
train_y = torch.cat([train_y1, train_y2])


# We will use the simplest form of GP model, exact inference
class ExactGPModel(models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distr.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)


smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 100

# model.covar_module.base_kernel.lengthscale = 0.01
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y) # +  (model.covar_module.base_kernel.lengthscale - 0.1) ** 2
    loss.backward()

    model.covar_module.base_kernel.lengthscale = 0.1

    if i % 5 == 4:
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
    optimizer.step()


# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.show()