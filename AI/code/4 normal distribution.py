import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, norm, expon, lognorm, gamma, beta, chi2

class norm1:
    def __init__(self, mean, sd, x):
        self.mean = mean
        self.sd = sd
        self.x = x
        
    def dist_curve(self):
        plt.plot(self.x, 1/(self.sd * np.sqrt(2 * np.pi)) *
                 np.exp( - (self.x - self.mean)**2 / (2 * self.sd**2) ), linewidth=2, color='y')

# پارامترهای مربوط به میانگین و انحراف معیار
mean1 = 5 
sd1 = 2

# تولید داده‌های تصادفی نرمال
c = np.random.normal(mean1, sd1, 3000)

# رسم هیستوگرام
count, bins, _ = plt.hist(c, bins=100, density=True, alpha=0.5, color='b')  # هیستوگرام بدون نرمال‌سازی

# محاسبه نقاط میانی بین bins برای رسم منحنی نرمال
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# رسم منحنی نرمال
hist1 = norm1(mean1, sd1, bin_centers)
hist1.dist_curve()

# نمایش نمودار
plt.title('Histogram and Normal Distribution Curve')
plt.show()

# محاسبه مقادیر پیش‌بینی شده برای توزیع نرمال
expected_counts = np.array([
    3000 * (norm.cdf(bins[i + 1], mean1, sd1) - norm.cdf(bins[i], mean1, sd1))
    for i in range(len(bins) - 1)
])

# نرمال‌سازی مقادیر پیش‌بینی‌شده برای تطابق با مجموع مقادیر مشاهده‌شده
expected_counts *= count.sum() / expected_counts.sum()

# آزمون کای-دو برای بررسی تطابق داده‌ها با توزیع نرمال
chi_statistic, p_value = chisquare(f_obs=count, f_exp=expected_counts)

# نمایش نتایج آزمون کای-دو
print("Chi square test:", chi_statistic)
print("p value:", p_value)

# لیست توزیع‌های مختلف برای برازش
distributions = [norm, expon, lognorm, gamma, beta, chi2]

# تابع برای محاسبه معیار برازش (کای-دو) برای هر توزیع
def fit_distribution(data, bins):
    best_p_value = 0
    best_distribution = None
    best_params = None
    
    for distribution in distributions:
        params = distribution.fit(data)
        expected_counts = np.array([
            3000 * (distribution.cdf(bins[i + 1], *params) - distribution.cdf(bins[i], *params))
            for i in range(len(bins) - 1)
        ])
        expected_counts *= count.sum() / expected_counts.sum()  # نرمال‌سازی
        chi_stat, p_value = chisquare(f_obs=count, f_exp=expected_counts)
        
        print(f"{distribution.name} - p_value: {p_value}")
        
        if p_value > best_p_value:
            best_p_value = p_value
            best_distribution = distribution
            best_params = params
            
    return best_distribution, best_params, best_p_value

# برازش توزیع‌های مختلف و انتخاب بهترین توزیع
best_distribution, best_params, best_p_value = fit_distribution(c, bins)

# نمایش بهترین توزیع
print(f"\nBest fitting distribution: {best_distribution.name}")
print(f"Best parameters: {best_params}")
print(f"Best p-value: {best_p_value}")

