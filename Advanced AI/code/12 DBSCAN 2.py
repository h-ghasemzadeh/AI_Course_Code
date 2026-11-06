# dbscan_clustering.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# ایجاد داده‌های نمونه با ساختار غیرکروی
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# اعمال الگوریتم DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# رسم نتایج خوشه‌بندی
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=30)
plt.title("DBSCAN Clustering Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
