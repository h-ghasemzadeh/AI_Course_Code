# gmm_clustering.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture

# تولید داده‌های مصنوعی به شکل نیم‌دایره
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# اعمال GMM با دو خوشه
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
labels = gmm.fit_predict(X)

# رسم نتایج خوشه‌بندی
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title("Gaussian Mixture Model (GMM) Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
