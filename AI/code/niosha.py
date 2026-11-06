import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, expon, lognorm, gamma,beta,chisquare
import pandas as pd
df1 = pd.read_csv(r"Iris.csv")
df2=pd.read_csv(r"Boston_Housing.csv")
'''
count, bins, ignored = plt.hist(df1["sepal_length"], bins=30, density=True, alpha=0.6, color='g')

mu, std = norm.fit(df1["sepal_length"])
loc0, scale0 = expon.fit(df1["sepal_length"])
print("Normal Distribution parameters:")
print("mean=",mu,",")     
print("standard deviation=" ,std)

print("Exponential Distribution parameters:" )
print("location=",loc0,",")     
print("scale=" ,scale0,)

bin_centers0 = 0.5 * (bins[1:] + bins[:-1])

# برازش توزیع نرمال
p_norm = norm.pdf(bin_centers0, mu, std)
plt.plot(bin_centers0, p_norm, 'k', linewidth=2, label='Normal fit')

# برازش توزیع نمایی
p_expon = expon.pdf(bin_centers0, loc0, scale0)
plt.plot(bin_centers0, p_expon, 'r', linewidth=2, label='Exponential fit')
plt.title("Histogram with Normal and Exponential Distribution Fits")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# sepal_width 
count, bins, ignored = plt.hist(df1["sepal_width"], bins=30, density=True, alpha=0.6, color='g')
shape1, loc1, scale1 = lognorm.fit(df1["sepal_width"])
shape2, loc2, scale2 = gamma.fit(df1["sepal_width"])

print("Log-Normal distribution parameters:" )
print("shape=",shape1,",") 
print("location=",loc1,",")
print("scale=" ,scale1)
print("Gamma Distribution parameters:" )
print("shape=",shape2,",") 
print("location=",loc2,",")
print("scale=" ,scale2)

# برازش توزیع لوگاریتم نرمال

bin_centers1= 0.5 * (bins[1:] + bins[:-1])
p1 = lognorm.pdf(bin_centers1, shape1, loc1, scale1)
plt.plot(bin_centers1, p1, 'r', linewidth=2, label='Log-Normal fit')

# برازش توزیع گاما
p2 = gamma.pdf(bin_centers1, shape2, loc2, scale2)
plt.plot(bin_centers1, p2, 'b', linewidth=2, label='Gamma fit')

plt.title("Histogram with Log-Normal and Gamma Distribution Fits")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
"هر 2 توزیع روی هم افتاده اند؟؟"
# # برازش توزیع لوگاریتم نرمال
# bin_centers0 = 0.5 * (bins[1:] + bins[:-1])
# p1 = lognorm.pdf(x, shape1, loc1, scale1)
# plt.plot(x, p1, 'r', linewidth=2, label='Log-Normal fit')

# # برازش توزیع گاما با داده‌های اولیه
# p2 = gamma.pdf(x, shape2, loc2, scale2)
# plt.plot(x, p2, 'b', linewidth=2, label='Gamma fit')

# # برازش مجدد توزیع گاما با تعدیل مکان و مقیاس
# shape2, loc2, scale2 = gamma.fit(df1["sepal_width"], floc=0)
# p2_refit = gamma.pdf(x, shape2, loc2, scale2)
# plt.plot(x, p2_refit, 'g--', linewidth=2, label='Gamma refit')

# plt.title("Histogram with Log-Normal and Gamma Distribution Fits")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.legend()
# plt.show()

# DIS 
count, bins, ignored = plt.hist(df2["DIS"], bins=30, density=True, alpha=0.6, color='g')
a, b, loc3, scale3 = beta.fit(df2["DIS"])
# a, b, loc3, scale3 = beta.fit(df2["DIS"], floc=0) # تنظیم loc به صفر
print("Beta distribution parameters:" )
print("alfa=",a,",") 
print("beta=",b,",")
print("location=",loc3,",")
print("scale=" ,scale3)
bin_centers3 = 0.5 * (bins[1:] + bins[:-1])
p3 = beta.pdf(bin_centers3, a, b, loc3, scale3)
plt.plot(bin_centers3, p3, 'r', linewidth=2, label='Beta fit')
plt.title("Histogram with Beta Distribution Fit")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Age
count, bins, ignored = plt.hist(df2["AGE"], bins=30, density=True, alpha=0.6, color='g')
df4, loc4, scale4 = chi2.fit(df2["AGE"])

print("Chi-Square distribution parameters:" )
print("Degrees of Freedom",df4,",") 
print("location=",loc4,",")
print("scale=" ,scale4)

# ایجاد نقاط میانی bins برای برازش توزیع
bin_centers4 = 0.5 * (bins[1:] + bins[:-1])

# برازش توزیع کی-اسکوئر بر اساس داده‌ها
p = chi2.pdf(bin_centers4, df4, loc4, scale4)
plt.plot(bin_centers4, p, 'r', linewidth=2, label='Chi-Square fit')

plt.title("Histogram with Chi-Square Distribution Fit")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
'''
'''
# از اینجا به بعد برای سوال دوم
from scipy.stats import norm, expon, lognorm, gamma, beta, chi2, chisquare

# توزیع‌هایی که می‌خواهیم بررسی کنیم
distributions = [norm, expon, lognorm, gamma, beta, chi2]

# تابع برای محاسبه معیار برازش (کای-دو) برای هر توزیع
def fit_distribution(data, bins, count):
    best_p_value = -1
    best_distribution = None
    best_params = None

    for distribution in distributions:
        try:
            params = distribution.fit(data)
            expected_counts = np.array([
                len(data) * (distribution.cdf(bins[i + 1], *params) - distribution.cdf(bins[i], *params))
                for i in range(len(bins) - 1)
            ])

            # نرمال‌سازی مقادیر مورد انتظار
            expected_counts *= count.sum() / expected_counts.sum()

            chi_stat, p_value = chisquare(f_obs=count, f_exp=expected_counts)

            if p_value > 0.05 :
                best_p_value = p_value
                best_distribution = distribution
                best_params = params
        except Exception as e:
            print(f"Error fitting {distribution.name}: {e}")

    return best_distribution, best_params, best_p_value


# بررسی توزیع‌ها برای ستون‌های مختلف
columns_to_check = [df1["sepal_length"], df1["sepal_width"], df2["DIS"], df2["AGE"]]
column_names = ["sepal_length", "sepal_width", "DIS", "AGE"]

for column, name in zip(columns_to_check, column_names):
    count, bins = np.histogram(column, bins=30, density=False)
    best_distribution, best_params, best_p_value = fit_distribution(column, bins, count)
    
    # بررسی بهترین توزیع
    if best_distribution:
        print(f"\nBest fitting distribution for {name}: {best_distribution.name}")
        print(f"Best parameters: {best_params}")
        print(f"Best p-value: {best_p_value}")
        
        # رسم بهترین توزیع برازش شده
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        pdf = best_distribution.pdf(bin_centers, *best_params)
        plt.hist(column, bins=30, density=True, alpha=0.6, color='g')
        plt.plot(bin_centers, pdf, 'r', linewidth=2, label=f'Best fit: {best_distribution.name}')
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Best fit for {name}")
        plt.show()
    else:
        print(f"No fitting distribution found for {name} with p-value > 0.05.")
        '''
distributions = [norm, expon, lognorm, gamma, beta, chi2]
count, bins, _ = plt.hist(df1['sepal_width'], bins=3, density=True, alpha=0.5, color='b')
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
        expected_counts *= count.sum() / expected_counts.sum() # نرمال‌سازی
        chi_stat, p_value = chisquare(f_obs=count, f_exp=expected_counts)
        
        print(f"{distribution.name} - p_value: {p_value}")
        
        if p_value > best_p_value:
            best_p_value = p_value
            best_distribution = distribution
            best_params = params
            
    return best_distribution, best_params, best_p_value

# برازش توزیع‌های مختلف و انتخاب بهترین توزیع
best_distribution, best_params, best_p_value = fit_distribution(df1['sepal_width'], bins)

# نمایش بهترین توزیع
print(f"\nBest fitting distribution: {best_distribution.name}")
print(f"Best parameters: {best_params}")
print(f"Best p-value: {best_p_value}")
