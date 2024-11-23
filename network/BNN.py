import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Bayesian Conv2D with Flipout
class BayesianConv2dFlipout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_sigma=1.0):
        super(BayesianConv2dFlipout, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights_mean = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.log_weights_sigma = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        
        self.bias_mean = nn.Parameter(torch.zeros(out_channels))
        self.log_bias_sigma = nn.Parameter(torch.zeros(out_channels))
        
    def sample_weights(self, stochastic=True):
        if stochastic:
            epsilon_w = Normal(torch.zeros_like(self.weights_mean), torch.ones_like(self.weights_mean)).sample()
            weights = self.weights_mean + torch.exp(self.log_weights_sigma) * epsilon_w
        else:
            weights = self.weights_mean
        return weights

    def sample_bias(self, stochastic=True):
        if stochastic:
            epsilon_b = Normal(torch.zeros_like(self.bias_mean), torch.ones_like(self.bias_mean)).sample()
            bias = self.bias_mean + torch.exp(self.log_bias_sigma) * epsilon_b
        else:
            bias = self.bias_mean
        return bias

    def forward(self, x, stochastic=True):
        weights = self.sample_weights(stochastic)
        bias = self.sample_bias(stochastic)
        return F.conv2d(x, weights, bias, stride=self.stride, padding=self.padding)


# Bayesian Dense with Flipout
class BayesianDenseFlipout(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super(BayesianDenseFlipout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights_mean = nn.Parameter(torch.zeros(out_features, in_features))
        self.log_weights_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.log_bias_sigma = nn.Parameter(torch.zeros(out_features))
        
    def sample_weights(self, stochastic=True):
        if stochastic:
            epsilon_w = Normal(torch.zeros_like(self.weights_mean), torch.ones_like(self.weights_mean)).sample()
            weights = self.weights_mean + torch.exp(self.log_weights_sigma) * epsilon_w
        else:
            weights = self.weights_mean
        return weights

    def sample_bias(self, stochastic=True):
        if stochastic:
            epsilon_b = Normal(torch.zeros_like(self.bias_mean), torch.ones_like(self.bias_mean)).sample()
            bias = self.bias_mean + torch.exp(self.log_bias_sigma) * epsilon_b
        else:
            bias = self.bias_mean
        return bias

    def forward(self, x, stochastic=True):
        weights = self.sample_weights(stochastic)
        bias = self.sample_bias(stochastic)
        return F.linear(x, weights, bias)


# Bayesian CNN model
class BayesianCNN(nn.Module):
    def __init__(self, prior_sigma=1.0):
        super(BayesianCNN, self).__init__()
        
        self.conv1 = BayesianConv2dFlipout(3, 8, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)
        self.conv2 = BayesianConv2dFlipout(8, 8, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)
        self.conv3 = BayesianConv2dFlipout(8, 16, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)
        self.conv4 = BayesianConv2dFlipout(16, 16, kernel_size=3, stride=1, padding=1, prior_sigma=prior_sigma)
        
        self.fc1 = BayesianDenseFlipout(16 * 8 * 8, 100, prior_sigma=prior_sigma)
        self.fc2 = BayesianDenseFlipout(100, 100, prior_sigma=prior_sigma)
        self.fc3 = BayesianDenseFlipout(100, 10, prior_sigma=prior_sigma)  # CIFAR10 has 10 classes

    def forward(self, x, stochastic=True):
        # Conv1 -> Conv2 -> MaxPool
        x = F.relu(self.conv1(x, stochastic))
        x = F.relu(self.conv2(x, stochastic))
        x = F.max_pool2d(x, 2)
        
        # Conv3 -> Conv4 -> MaxPool
        x = F.relu(self.conv3(x, stochastic))
        x = F.relu(self.conv4(x, stochastic))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x, stochastic))
        x = F.relu(self.fc2(x, stochastic))
        x = self.fc3(x, stochastic)
        
        return F.log_softmax(x, dim=-1)