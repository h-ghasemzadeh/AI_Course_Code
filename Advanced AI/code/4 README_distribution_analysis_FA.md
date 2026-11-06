# تحلیل توزیع آماری داده‌های مهندسی عمران

## داده واقعی
داده پیشنهادی: **Concrete Compressive Strength** از مخزن UCI؛ شامل ۱۰۳۰ رکورد و ۹ ستون (۸ متغیر طرح اختلاط/سن و مقاومت فشاری اندازه‌گیری‌شده بتن بر حسب MPa). داده اصلی فاقد مقادیر گمشده گزارش شده و مجوز آن CC BY 4.0 است.

- صفحه و دانلود رسمی: https://archive.ics.uci.edu/dataset/165/concrete-compressive-strength
- استناد: Yeh, I. (1998). *Concrete Compressive Strength*. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67

## اجرا
در ترمینال:
```bash
pip install numpy pandas scipy matplotlib openpyxl ucimlrepo
python distribution_analysis.py
```
اجرای بالا داده UCI را از اینترنت دریافت می‌کند. اگر فایل Excel یا CSV را جداگانه از صفحه UCI دانلود کردید:
```bash
python distribution_analysis.py --file Concrete_Data.xls --column "Concrete compressive strength"
```
نام دقیق ستون را می‌توانید با اجرای زیر مشاهده کنید:
```bash
python distribution_analysis.py --file Concrete_Data.xls
```
اگر نام ستون را ندهید، برنامه فهرست ستون‌ها را چاپ می‌کند و متوقف می‌شود؛ سپس نام درست را وارد کنید.

## خروجی‌ها
- `descriptive_statistics.csv`: میانگین، واریانس نمونه، انحراف معیار، چولگی، ضریب تغییرات، چارک‌ها و کشیدگی
- `continuous_distribution_fits.csv`: پارامترهای برازش‌شده نرمال، لگ‌نرمال، گاما، بتا، کای‌دو و نمایی؛ درست‌نمایی، AIC/BIC و آزمون‌های برازش در صورت پشتیبانی SciPy
- `discrete_distribution_fits_rounded_data.csv`: پواسون و دوجمله‌ای روی مقادیر گرد‌شده، به‌صورت تحلیل جداگانه
- `normality_tests.csv`: شاپیرو–ویلک، دآگوستینو–پیرسون، جارک–برا و اندرسون–دارلینگ برای نرمال‌بودن
- `01_data_and_histogram.png`, `02_qq_plots.png`, `03_ecdf_fitted_cdfs.png`: نمودار داده، هیستوگرام با PDFهای برازش‌شده، نمودارهای Q–Q و CDF

## نکات آماری مهم
1. مقاومت فشاری بتن یک متغیر پیوسته است؛ بنابراین پواسون و دوجمله‌ای مستقیماً برای داده خام MPa مناسب نیستند. برنامه آن‌ها را فقط برای داده‌های گرد‌شده و در جدول جداگانه بررسی می‌کند.
2. آزمون‌های Shapiro–Wilk، D’Agostino–Pearson و Jarque–Bera آزمون نرمال‌بودن‌اند، نه آزمون عمومی همه توزیع‌ها.
3. در صورت وجود نسخه سازگار SciPy، p-value آزمون‌های KS و AD با شبیه‌سازی مونت‌کارلو و برازش مجدد پارامترها محاسبه می‌شود؛ این روش از p-value ساده KS پس از تخمین پارامترها مناسب‌تر است. آزمون کای‌دو گروه‌بندی‌شده تقریبی است.
4. برای انتخاب توزیع، تنها به p-value تکیه نکنید: AIC/BIC، نمودار Q–Q، شکل دنباله‌ها، دامنه مجاز فیزیکی و هدف تحلیل را نیز مقایسه کنید. «بزرگ‌ترین p-value» به‌تنهایی به معنای بهترین توزیع قطعی نیست.
