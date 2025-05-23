# Problem 1: The researcher is interested in  evaluating public trust in a specific news source.
# Each person in a sample rates their level of trust in the news source on a scale from 0 (no trust at all) to 10 (complete trust).
# The researcher want to use Bayesian inference to estimate:
# The average trust score (μ) among the sample population.
# The variability (σ) in trust across individuals.

import numpy as np
import matplotlib.pyplot as plt

# Trust scores from a normal distribution
true_mu = 6.5  # True mu trust in the news source
true_sigma = 1.2  # True std in trust
data = np.random.normal(true_mu, true_sigma, size=100)
# Prior parameters
prior_mu = 5.0  # Belief about average trust
prior_sigma = 2.0  # Uncertainty in prior belief
prior_precision = 1 / (prior_sigma ** 2)
# Prior for sigma
prior_shape = 2
prior_rate = 1
# Calculating the Posterior
sample_mean = np.mean(data)
sample_size = len(data)
posterior_mu_precision = prior_precision + sample_size
posterior_mu_mean = (prior_mu * prior_precision + sample_mean * sample_size) / posterior_mu_precision
posterior_mu_std = np.sqrt(1 / posterior_mu_precision)
# Posterior of mu
posterior_mu_samples = np.random.normal(posterior_mu_mean, posterior_mu_std, size=10000)
# Posterior for sigma
posterior_shape = prior_shape + sample_size / 2
posterior_rate = prior_rate + 0.5 * np.sum((data - sample_mean) ** 2)
posterior_sigma_samples = np.sqrt(1 / np.random.gamma(posterior_shape, 1 / posterior_rate, size=10000))
# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu_samples, bins=40, density=True, color='pink', edgecolor='black')
plt.title("Posterior distribution of μ")
plt.xlabel("μ (Average trust score)")
plt.ylabel("Density")
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=40, density=True, color='orange', edgecolor='black')
plt.title("Posterior distribution of σ")
plt.xlabel("σ (Trust score variability)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
# Summary statistics
mean_mu = np.mean(posterior_mu_samples)
std_mu = np.std(posterior_mu_samples)
print(f"Posterior mean of μ (average trust): {mean_mu:.2f}")
print(f"Standard deviation of μ: {std_mu:.2f}")
