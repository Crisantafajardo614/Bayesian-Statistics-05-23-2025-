# The researcher is assessing how much misinformation a particular influencer spreads.
# Experts or algorithms assign a misinformation score to each post the influencer shares, ranging from 0 (no misinformation) to 10 (severe misinformation).
# The goal is to estimate:
# The average misinformation level (μ) per post.
# The inconsistency or variability (σ) in misinformation across the influencer’s content.

import numpy as np
import matplotlib.pyplot as plt

# Misinformation scores for influencer's posts
true_mu = 4.0  # Average misinformation level
true_sigma = 2.0  # Variability in misinformation
data = np.random.normal(true_mu, true_sigma, size=100)
# Prior beliefs
prior_mu = 5.0  # Initial belief about average misinformation
prior_sigma = 3.0  # Uncertainty in prior
prior_precision = 1 / (prior_sigma ** 2)
# Prior for sigma
prior_shape = 2
prior_rate = 1
# Posterior calculations
sample_mean = np.mean(data)
sample_size = len(data)
posterior_mu_precision = prior_precision + sample_size
posterior_mu_mean = (prior_mu * prior_precision + sample_mean * sample_size) / posterior_mu_precision
posterior_mu_std = np.sqrt(1 / posterior_mu_precision)
# Posterior of μ
posterior_mu_samples = np.random.normal(posterior_mu_mean, posterior_mu_std, size=10000)
# Posterior for σ using inverse gamma
posterior_shape = prior_shape + sample_size / 2
posterior_rate = prior_rate + 0.5 * np.sum((data - sample_mean) ** 2)
posterior_sigma_samples = np.sqrt(1 / np.random.gamma(posterior_shape, 1 / posterior_rate, size=10000))
# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu_samples, bins=40, density=True, color='red', edgecolor='black')
plt.title("Posterior distribution of μ")
plt.xlabel("μ (Average misinformation score)")
plt.ylabel("Density")
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=40, density=True, color='gray', edgecolor='black')
plt.title("Posterior distribution of σ")
plt.xlabel("σ (Misinformation variability)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
# Summary statistics
mean_mu = np.mean(posterior_mu_samples)
std_mu = np.std(posterior_mu_samples)
print(f"Posterior mean of μ (average misinformation): {mean_mu:.2f}")
print(f"Standard deviation of μ: {std_mu:.2f}")
