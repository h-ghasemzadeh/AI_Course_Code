import pandas as pd
from scipy import stats

# خواندن فایل داده
try:
    data = pd.read_csv('sugar_cookie.csv')
    print("داده‌ها با موفقیت خوانده شدند:")
    print(data.head())
except FileNotFoundError:
    print("خطا: فایل sugar_cookie.csv یافت نشد.")
    exit()

# نمایش ستون‌های موجود
print("\nستون‌های موجود در داده‌ها:")
print(data.columns)

# انتخاب ستون برای تحلیل
column_name = input("\nنام ستون مورد نظر برای آزمون t تک نمونه‌ای را وارد کنید: ")

try:
    sample_data = data[column_name].dropna()  # حذف مقادیر缺失
    if len(sample_data) == 0:
        print("خطا: ستون انتخاب شده فاقد داده است.")
        exit()
    
    # مقدار مورد آزمون
    test_value = 10
    
    # انجام آزمون t تک نمونه‌ای
    t_stat, p_value = stats.ttest_1samp(sample_data, popmean=test_value)
    
    print("\nنتایج آزمون t تک نمونه‌ای:")
    print(f"میانگین نمونه: {sample_data.mean():.4f}")
    print(f"مقدار مورد آزمون: {test_value}")
    print(f"تعداد نمونه‌ها: {len(sample_data)}")
    print(f"آماره t: {t_stat:.4f}")
    print(f"مقدار p دوطرفه: {p_value:.4f}")
    
    # سطح معنی‌داری
    alpha = 0.05
    
    print(f"\nسطح معنی‌داری: {alpha}")
    
    if p_value < alpha:
        print(f"نتیجه: میانگین داده‌ها به طور معنی‌داری با {test_value} متفاوت است (p < {alpha})")
    else:
        print(f"نتیجه: میانگین داده‌ها به طور معنی‌داری با {test_value} متفاوت نیست (p ≥ {alpha})")
    
    # محاسبه فاصله اطمینان 95%
    confidence_level = 0.95
    degrees_freedom = len(sample_data) - 1
    sample_mean = sample_data.mean()
    sample_std = sample_data.std(ddof=1)
    confidence_interval = stats.t.interval(
        confidence_level, 
        degrees_freedom, 
        loc=sample_mean, 
        scale=sample_std/(len(sample_data)**0.5)
    )
    
    print(f"\nفاصله اطمینان {confidence_level*100:.0f}% برای میانگین:")
    print(f"از {confidence_interval[0]:.4f} تا {confidence_interval[1]:.4f}")
    
except KeyError:
    print("خطا: نام ستون وارد شده صحیح نیست.")
except Exception as e:
    print(f"خطای غیرمنتظره: {e}")