import torch
from torch import Tensor
import math
import numpy as np

"""

Exercise 1.1: Diagonal Gaussian Likelihood

Write a function that takes in PyTorch Tensors for the means and 
log stds of a batch of diagonal Gaussian distributions, along with a 
PyTorch Tensor for (previously-generated) samples from those 
distributions, and returns a Tensor containing the log 
likelihoods of those samples.

"""


def gaussian_likelihood(x: Tensor, mu: Tensor, log_std: Tensor):
    """
    Gaussian Likelihood

    Based on the formula at
    https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies

    Args:
        x: Tensor with shape [batch, dim] (sample)
        mu: Tensor with shape [batch, dim] (means)
        log_std: Tensor with shape [batch, dim] or [dim] (log std. either as a
        single vector not dependent on state, or as a matrix with one value per
        dimension e.g. from a NN)

    Returns:
        Tensor with shape [batch], with the log likelihoods for each sample
    """

    # Get the number of dimensions
    k = x.size()[1]

    # Create end term = k log(2 pi)
    k_log_2_pi = k * math.log(2 * math.pi)

    # Create variance values
    variance = log_std.exp().pow(2)

    # Create sum term
    inside_sum_all_k = (x - mu).pow(2) / variance + 2 * log_std
    sum_all_k = torch.sum(inside_sum_all_k, dim=1)

    # Put it all together
    res = -0.5 * (sum_all_k + k_log_2_pi)

    return res


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """
    from spinup.exercises.pytorch.problem_set_1_solutions import exercise1_1_soln
    from spinup.exercises.common import print_result

    batch_size = 32
    dim = 10

    x = torch.rand(batch_size, dim)
    mu = torch.rand(batch_size, dim)
    log_std = torch.rand(dim)

    your_gaussian_likelihood = gaussian_likelihood(x, mu, log_std)
    true_gaussian_likelihood = exercise1_1_soln.gaussian_likelihood(
        x, mu, log_std)

    your_result = your_gaussian_likelihood.detach().numpy()
    true_result = true_gaussian_likelihood.detach().numpy()

    correct = np.allclose(your_result, true_result)
    print_result(correct)
