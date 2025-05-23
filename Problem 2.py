# The researcher is analyzing a set of social media posts to detect how “fake” or misleading they are.
# A machine learning model gives each post a fake score from 0 (completely authentic) to 10 (extremely fake).
# The researcher wants to estimate:
# The average fake score (μ) in the sample.
# The variability (σ) in how fake the posts are.

import numpy as np
import matplotlib.pyplot as plt

# Fake scores from social media posts
true_mu = 6.0  # true mean fake score
true_sigma = 1.5  # true std of fake scores
data = np.random.normal(true_mu, true_sigma, size=100)
# Prior beliefs
prior_mu = 5.0  # prior belief about average fakeness
prior_sigma = 2.0  # uncertainty in prior belief
prior_precision = 1 / (prior_sigma ** 2)
# Prior for sigma
prior_shape = 2
prior_rate = 1
# Calculations of the Posterior
sample_mean = np.mean(data)
sample_size = len(data)
posterior_mu_precision = prior_precision + sample_size
posterior_mu_mean = (prior_mu * prior_precision + sample_mean * sample_size) / posterior_mu_precision
posterior_mu_std = np.sqrt(1 / posterior_mu_precision)
# Posterior for μ
posterior_mu_samples = np.random.normal(posterior_mu_mean, posterior_mu_std, size=10000)
# Posterior for σ
posterior_shape = prior_shape + sample_size / 2
posterior_rate = prior_rate + 0.5 * np.sum((data - sample_mean) ** 2)
posterior_sigma_samples = np.sqrt(1 / np.random.gamma(posterior_shape, 1 / posterior_rate, size=10000))
# Plotting results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu_samples, bins=40, density=True, color='orange', edgecolor='black')
plt.title("Posterior distribution of μ")
plt.xlabel("μ (Average fake score)")
plt.ylabel("Density")
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=40, density=True, color='purple', edgecolor='black')
plt.title("Posterior distribution of σ")
plt.xlabel("σ (Fake score variability)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
# Summary statistics
mean_mu = np.mean(posterior_mu_samples)
std_mu = np.std(posterior_mu_samples)
print(f"Posterior mean of μ (average fake score): {mean_mu:.2f}")
print(f"Standard deviation of μ: {std_mu:.2f}")
