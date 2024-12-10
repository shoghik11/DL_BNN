import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def kl_divergence(mean, log_sigma, prior_sigma):
    """
    Compute the KL divergence between a Gaussian posterior and a Gaussian prior.

    Parameters:
    - mean: Posterior mean tensor.
    - log_sigma: Logarithm of the posterior standard deviation tensor.
    - prior_sigma: Standard deviation of the Gaussian prior (scalar).

    Returns:
    - KL divergence (scalar).
    """
    sigma = torch.exp(log_sigma)
    prior_sigma = torch.tensor(prior_sigma, device=mean.device)  # Convert prior_sigma to a tensor

    prior_var = prior_sigma ** 2
    posterior_var = sigma ** 2

    kl = 0.5 * (
        posterior_var / prior_var +
        (mean ** 2) / prior_var -
        1 +
        2 * log_sigma -
        2 * torch.log(prior_sigma)
    )
    return kl.sum()



# Bayesian Conv2D with Flipout
class BayesianConv2dFlipout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_sigma=1.0):
        super(BayesianConv2dFlipout, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma

        self.weights_mean = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.log_weights_sigma = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))

        self.bias_mean = nn.Parameter(torch.zeros(out_channels))
        self.log_bias_sigma = nn.Parameter(torch.zeros(out_channels))

        # Initialize parameters
        nn.init.xavier_uniform_(self.weights_mean)
        nn.init.constant_(self.log_weights_sigma, -3.0)
        nn.init.constant_(self.bias_mean, 0)
        nn.init.constant_(self.log_bias_sigma, -3.0)

    def sample_weights(self, stochastic=True):
        if stochastic:
            epsilon_w = Normal(0, 1).sample(self.weights_mean.size()).to(self.weights_mean.device)
            weights = self.weights_mean + torch.exp(self.log_weights_sigma) * epsilon_w
        else:
            weights = self.weights_mean
        return weights

    def sample_bias(self, stochastic=True):
        if stochastic:
            epsilon_b = Normal(0, 1).sample(self.bias_mean.size()).to(self.bias_mean.device)
            bias = self.bias_mean + torch.exp(self.log_bias_sigma) * epsilon_b
        else:
            bias = self.bias_mean
        return bias

    def forward(self, x, stochastic=True):
        weights = self.sample_weights(stochastic)
        bias = self.sample_bias(stochastic)
        return F.conv2d(x, weights, bias, stride=self.stride, padding=self.padding)

    def kl_loss(self):
        return kl_divergence(self.weights_mean, self.log_weights_sigma, self.prior_sigma) + \
               kl_divergence(self.bias_mean, self.log_bias_sigma, self.prior_sigma)


# Bayesian Dense with Flipout
class BayesianDenseFlipout(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super(BayesianDenseFlipout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weights_mean = nn.Parameter(torch.zeros(out_features, in_features))
        self.log_weights_sigma = nn.Parameter(torch.zeros(out_features, in_features))

        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.log_bias_sigma = nn.Parameter(torch.zeros(out_features))

        # Initialize parameters
        nn.init.xavier_uniform_(self.weights_mean)
        nn.init.constant_(self.log_weights_sigma, -3.0)
        nn.init.constant_(self.bias_mean, 0)
        nn.init.constant_(self.log_bias_sigma, -3.0)

    def sample_weights(self, stochastic=True):
        if stochastic:
            epsilon_w = Normal(0, 1).sample(self.weights_mean.size()).to(self.weights_mean.device)
            weights = self.weights_mean + torch.exp(self.log_weights_sigma) * epsilon_w
        else:
            weights = self.weights_mean
        return weights

    def sample_bias(self, stochastic=True):
        if stochastic:
            epsilon_b = Normal(0, 1).sample(self.bias_mean.size()).to(self.bias_mean.device)
            bias = self.bias_mean + torch.exp(self.log_bias_sigma) * epsilon_b
        else:
            bias = self.bias_mean
        return bias

    def forward(self, x, stochastic=True):
        weights = self.sample_weights(stochastic)
        bias = self.sample_bias(stochastic)
        return F.linear(x, weights, bias)

    def kl_loss(self):
        return kl_divergence(self.weights_mean, self.log_weights_sigma, self.prior_sigma) + \
               kl_divergence(self.bias_mean, self.log_bias_sigma, self.prior_sigma)


# Bayesian CNN model
class BayesianCNN(nn.Module):
    def __init__(self, prior_sigma=1.0):
        super().__init__()  # Corrected this line
        self.conv1 = BayesianConv2dFlipout(3, 8, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)
        self.conv2 = BayesianConv2dFlipout(8, 8, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)
        self.conv3 = BayesianConv2dFlipout(8, 16, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)
        self.conv4 = BayesianConv2dFlipout(16, 16, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)

        self.fc1 = BayesianDenseFlipout(16 * 8 * 8, 100, prior_sigma=prior_sigma)
        self.fc2 = BayesianDenseFlipout(100, 100, prior_sigma=prior_sigma)
        self.fc3 = BayesianDenseFlipout(100, 10, prior_sigma=prior_sigma)  # CIFAR-10 has 10 classes

    def forward(self, x, stochastic=True):
        # Conv1 -> Conv2 -> MaxPool
        x = F.relu(self.conv1(x, stochastic))
        x = F.relu(self.conv2(x, stochastic))
        x = F.max_pool2d(x, 2)

        # Conv3 -> Conv4 -> MaxPool
        x = F.relu(self.conv3(x, stochastic))
        x = F.relu(self.conv4(x, stochastic))
        x = F.max_pool2d(x, 2)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x, stochastic))
        x = F.relu(self.fc2(x, stochastic))
        x = self.fc3(x, stochastic)

        return F.log_softmax(x, dim=-1)

    def kl_loss(self):
        return sum(layer.kl_loss() for layer in self.modules() if isinstance(layer, (BayesianConv2dFlipout, BayesianDenseFlipout)))
