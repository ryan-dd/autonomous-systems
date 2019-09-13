import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from math import sqrt


def create_plotting_parameters_for_normal_distribution(mean, variance, num_std_dev_to_plot=3):
    """Create parameters to easily plot a normal distribution

    Arguments:
        mean {float} -- Mean of the normal distribution
        variance {float} -- Variance of the normal distribution

    """
    sigma = sqrt(variance)
    # I set the default range of x to be plus or minus 3 sigma since that has
    # the important visual part of the normal distribution
    x = np.linspace(mean - num_std_dev_to_plot*sigma,
                    mean + num_std_dev_to_plot*sigma, 100)
    y = stats.norm.pdf(x, mean, sigma)
    return x, y


if __name__ == "__main__":
    x, y = create_plotting_parameters_for_normal_distribution(0, 3)
    plt.plot(x, y)
    plt.show()
