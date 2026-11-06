import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import arabic_reshaper
from bidi.algorithm import get_display

# مسیر فونت فارسی
font_path = r'C:\lecture\AI\support\fonts\Vazirmatn-VariableFont_wght.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Vazirmatn'
plt.rcParams['axes.unicode_minus'] = False

# تابع کمکی برای اصلاح متن فارسی
def farsi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)  # اتصال حروف
    bidi_text = get_display(reshaped_text)          # راست‌به‌چپ کردن
    return bidi_text

# داده‌ها (مثال از زلزله‌ها)
labels = [
    "۱۳۰۲ کاشمر", "۱۳۳۶ سنگچال", "۱۳۴۱ بوئین‌زهرا", "۱۳۵۷ طبس",
    "۱۳۶۹ رودبار", "۱۳۸۲ بم", "۱۳۹۶ ازگله"
]
values = [770, 1200, 11000, 20000, 37000, 50000, 620]

# اعمال اصلاحات فارسی روی برچسب‌ها
labels = [farsi_text(l) for l in labels]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, values, color='skyblue', edgecolor='black')

plt.title(farsi_text("تلفات زلزله‌های بزرگ ایران (۱۳۰۰ تا ۱۴۰۰)"), fontsize=14)
plt.xlabel(farsi_text("سال و نام زلزله"), fontsize=12)
plt.ylabel(farsi_text("تعداد کشته‌شدگان"), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
