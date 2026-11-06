import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# تولید داده‌های مصنوعی
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# مقادیر مختلف برای عمق درخت
depths = range(1, 21)

# لیست برای ذخیره خطاها
train_errors = []
test_errors = []

# آموزش مدل با عمق‌های مختلف
for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    # پیش‌بینی
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # محاسبه خطا
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# رسم نمودار خطاها
plt.figure(figsize=(10, 6))
plt.plot(depths, train_errors, label="Training Error", marker='o')
plt.plot(depths, test_errors, label="Testing Error", marker='o')
plt.xlabel("Tree Depth")
plt.ylabel("Mean Squared Error")
plt.title("Effect of Tree Depth on Variance")
plt.legend()
plt.grid()
plt.show()
