"""
کپچا فارسی پیشرفته – Human-readable, AI-resistant
====================================================
تکنیک‌های اصلی برای مقاومت در برابر AI:
  1. Elastic distortion  – کشش موضعی تصادفی روی پیکسل‌های متن
  2. Occluding bands     – نوارهای رنگی نیمه‌شفاف که متن را قطع می‌کنند
  3. Fake ligatures      – خطوط مصنوعی که کلمات را به هم وصل می‌کنند
  4. Fragment noise      – اجزای شبیه حروف فارسی به عنوان نویز
  5. Per-pixel hue shift – تغییر رنگ پیکسل‌های متن به صورت موضعی
  6. Wave + perspective  – تابع sin روی کل تصویر

نصب وابستگی‌ها:
  pip install arabic-reshaper python-bidi pillow numpy scipy
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random, os, glob, math
from scipy.ndimage import map_coordinates, gaussian_filter

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RESHAPER_AVAILABLE = True
except ImportError:
    RESHAPER_AVAILABLE = False
    print("⚠️  pip install arabic-reshaper python-bidi")


# ─────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────

def is_persian_word(word: str) -> bool:
    """تشخیص اینکه کلمه فارسی است یا انگلیسی"""
    # حروف فارسی در محدوده U+0600 تا U+06FF قرار دارند
    persian_chars = 0
    for char in word:
        if '\u0600' <= char <= '\u06FF':
            persian_chars += 1
    # اگر بیش از 50% حروف فارسی باشد، کلمه فارسی محسوب می‌شود
    return persian_chars > len(word) * 0.5


def reshape_word(word: str) -> str:
    """reshape کردن کلمه اگر فارسی باشد"""
    if not RESHAPER_AVAILABLE:
        return word
    
    # اگر کلمه فارسی نیست، بدون تغییر برگردان
    if not is_persian_word(word):
        return word
    
    try:
        reshaped = arabic_reshaper.reshape(word)
        return get_display(reshaped)
    except Exception:
        return word


def get_text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        return len(text) * 22, 36


# ─────────────────────────────────────────────
#  Core distortions
# ─────────────────────────────────────────────

def elastic_distort(img_arr: np.ndarray, alpha: float = 28, sigma: float = 5) -> np.ndarray:
    """کشش الاستیک تصادفی – ساختار را می‌شکند ولی برای انسان خوانا می‌ماند"""
    h, w = img_arr.shape[:2]
    rng = np.random.RandomState(random.randint(0, 99999))
    dx = gaussian_filter(rng.rand(h, w) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(rng.rand(h, w) * 2 - 1, sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    xmap = np.clip(x + dx, 0, w - 1)
    ymap = np.clip(y + dy, 0, h - 1)

    out = np.zeros_like(img_arr)
    for c in range(img_arr.shape[2]):
        out[:, :, c] = map_coordinates(img_arr[:, :, c], [ymap.ravel(), xmap.ravel()],
                                       order=1, mode='nearest').reshape(h, w)
    return out.astype(np.uint8)


def wave_warp(img: Image.Image, amp_x: int = 5, amp_y: int = 4) -> Image.Image:
    """تابع موجی روی کل تصویر"""
    arr = np.array(img)
    h, w = arr.shape[:2]
    freq = random.uniform(0.018, 0.035)
    phase = random.uniform(0, 2 * math.pi)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    xmap = np.clip(x + amp_x * np.sin(2 * math.pi * y * freq + phase), 0, w - 1)
    ymap = np.clip(y + amp_y * np.sin(2 * math.pi * x * freq + phase), 0, h - 1)

    out = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        out[:, :, c] = map_coordinates(arr[:, :, c], [ymap.ravel(), xmap.ravel()],
                                       order=1, mode='nearest').reshape(h, w)
    return Image.fromarray(out.astype(np.uint8))


def add_occluding_bands(img: Image.Image, n: int = 4) -> Image.Image:
    """نوارهای رنگی که از روی متن رد می‌شوند – OCR را گیج می‌کنند"""
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    for _ in range(n):
        x1 = random.randint(-30, w // 2)
        y1 = random.randint(20, h - 40)
        x2 = random.randint(w // 2, w + 30)
        y2 = y1 + random.randint(8, 20)
        alpha_val = random.randint(60, 140)
        color = (random.randint(80, 200),
                 random.randint(80, 200),
                 random.randint(80, 200),
                 alpha_val)
        draw.rectangle([x1, y1, x2, y2], fill=color)
    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


def add_fake_ligatures(img: Image.Image, word_positions: list) -> Image.Image:
    """خطوط کوتاه مصنوعی بین کلمات – تقسیم‌بندی AI را مختل می‌کند"""
    draw = ImageDraw.Draw(img)
    for i in range(len(word_positions) - 1):
        x1, y1 = word_positions[i]
        x2, y2 = word_positions[i + 1]
        mid_x = (x1 + x2) // 2 + random.randint(-15, 15)
        mid_y = (y1 + y2) // 2 + random.randint(-8, 8)
        color = (random.randint(50, 150),
                 random.randint(50, 150),
                 random.randint(50, 150))
        draw.line([(x1, y1 + random.randint(-5, 5)),
                   (mid_x, mid_y),
                   (x2, y2 + random.randint(-5, 5))],
                  fill=color, width=random.randint(1, 3))
    return img


def add_fragment_noise(img: Image.Image, n: int = 30) -> Image.Image:
    """اجزای شبیه حروف فارسی به عنوان نویز"""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for _ in range(n):
        x = random.randint(0, w)
        y = random.randint(30, h - 30)
        color = (random.randint(100, 200),
                 random.randint(100, 200),
                 random.randint(100, 200))
        kind = random.choice(['arc', 'dot_cluster', 'short_curve', 'zigzag'])

        if kind == 'arc':
            r = random.randint(6, 18)
            start_a = random.randint(0, 360)
            end_a = start_a + random.randint(60, 200)
            draw.arc([x - r, y - r, x + r, y + r], start_a, end_a,
                     fill=color, width=random.randint(1, 3))

        elif kind == 'dot_cluster':
            for _ in range(random.randint(2, 4)):
                dx, dy = random.randint(-10, 10), random.randint(-5, 5)
                r = random.randint(2, 5)
                draw.ellipse([x + dx - r, y + dy - r, x + dx + r, y + dy + r], fill=color)

        elif kind == 'short_curve':
            pts = [(x + i * 6 + random.randint(-3, 3),
                    y + int(8 * math.sin(i * 0.8)) + random.randint(-4, 4))
                   for i in range(random.randint(3, 7))]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=random.randint(1, 3))

        else:  # zigzag
            pts = [(x + i * 7, y + (0 if i % 2 == 0 else random.randint(8, 16)))
                   for i in range(random.randint(3, 6))]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)

    return img


def pixel_hue_noise(img: Image.Image, strength: int = 18) -> Image.Image:
    """تغییر رنگ موضعی روی پیکسل‌های تیره (متن) – OCR color را گیج می‌کند"""
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-strength, strength + 1, arr.shape)
    dark_mask = arr.mean(axis=2) < 160
    for c in range(3):
        arr[:, :, c] = np.where(dark_mask, arr[:, :, c] + noise[:, :, c], arr[:, :, c])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def perspective_skew(img: Image.Image, max_skew: float = 0.06) -> Image.Image:
    """کج‌شدگی خفیف perspective – feature extraction را مختل می‌کند"""
    w, h = img.size
    dx = int(w * random.uniform(-max_skew, max_skew))
    dy = int(h * random.uniform(-max_skew, max_skew))
    coeffs = (1, dx / h, 0,
              dy / w, 1, 0,
              random.uniform(-0.0002, 0.0002),
              random.uniform(-0.0002, 0.0002))
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


# ─────────────────────────────────────────────
#  Background
# ─────────────────────────────────────────────

def create_background(width: int, height: int) -> Image.Image:
    bg_choices = [
        (245, 240, 228),
        (235, 238, 245),
        (240, 245, 235),
    ]
    bg_color = random.choice(bg_choices)
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    for _ in range(random.randint(18, 28)):
        x1 = random.randint(-60, width + 60)
        y1 = random.randint(-60, height + 60)
        angle = random.uniform(-70, 70)
        length = random.randint(180, width + 200)
        x2 = x1 + length * math.cos(math.radians(angle))
        y2 = y1 + length * math.sin(math.radians(angle))
        color = tuple(random.randint(150, 210) for _ in range(3))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 5))

    for _ in range(random.randint(6, 12)):
        pts = [(random.randint(-20, width + 20), random.randint(-20, height + 20))
               for _ in range(random.randint(10, 18))]
        color = tuple(random.randint(150, 210) for _ in range(3))
        draw.line(pts, fill=color, width=random.randint(1, 4))

    for _ in range(random.randint(4, 10)):
        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(5, 22)
        color = tuple(random.randint(160, 215) for _ in range(3))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=random.randint(1, 3))

    return img


# ─────────────────────────────────────────────
#  Font helpers
# ─────────────────────────────────────────────

FONT_NAMES = [
    "Vazir.ttf", "Samim.ttf", "Shabnam.ttf", "Gandom.ttf",
    "Tanha.ttf", "IRANSans.ttf", "BNAzarin.ttf",
    "Yagut.ttf", "Sahel.ttf", "Parastoo.ttf",
    "Roya.ttf", "Mitra.ttf",
    "Arial.ttf", "Times New Roman.ttf", "Calibri.ttf"  # فونت‌های انگلیسی
]

SEARCH_DIRS = [
    "C:/Windows/Fonts/",
    "/usr/share/fonts/truetype/",
    "/System/Library/Fonts/",
    "./fonts/",
]


def find_fonts() -> list:
    found = []
    for d in SEARCH_DIRS:
        if not os.path.exists(d):
            continue
        for name in FONT_NAMES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                found.append(p)
    found = list(dict.fromkeys(found))
    if not found:
        os.makedirs("./fonts/", exist_ok=True)
        print("📁 پوشه ./fonts/ را با فونت‌های فارسی پر کنید.")
    return found


def get_font(paths: list, size: int, is_english: bool = False):
    if not paths:
        return ImageFont.load_default()
    
    # اگر کلمه انگلیسی است، فونت انگلیسی انتخاب کن
    if is_english:
        english_fonts = ["Arial.ttf", "Times New Roman.ttf", "Calibri.ttf"]
        for font_name in english_fonts:
            for path in paths:
                if font_name in path:
                    try:
                        return ImageFont.truetype(path, size)
                    except:
                        continue
    
    try:
        return ImageFont.truetype(random.choice(paths), size)
    except Exception:
        return ImageFont.load_default()

def auto_crop(image: Image.Image) -> Image.Image:
    """برش خودکار برای حذف حاشیه‌های سیاه یا سفید"""
    # تبدیل به آرایه
    img_array = np.array(image)
    
    # پیدا کردن پیکسل‌های غیر خالی (غیر سیاه)
    # اگر مقدار متوسط کمتر از 20 باشد، سیاه محسوب می‌شود
    non_empty = np.where(img_array.mean(axis=2) > 20)
    
    if len(non_empty[0]) == 0 or len(non_empty[1]) == 0:
        return image
    
    # پیدا کردن مرزهای محتوا
    y_min = np.min(non_empty[0])
    y_max = np.max(non_empty[0])
    x_min = np.min(non_empty[1])
    x_max = np.max(non_empty[1])
    
    # اضافه کردن حاشیه کوچک
    padding = 10
    y_min = max(0, y_min - padding)
    y_max = min(image.height, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.width, x_max + padding)
    
    # برش تصویر
    return image.crop((x_min, y_min, x_max, y_max))
# ─────────────────────────────────────────────
#  Main CAPTCHA class
# ─────────────────────────────────────────────

class PersianCAPTCHAAdvanced:

    ELASTIC_ALPHA       = 3
    ELASTIC_SIGMA       = 1.5
    WAVE_AMP_X          = 1.5
    WAVE_AMP_Y          = 3
    OCCLUDING_BANDS     = 4
    FAKE_LIGATURES      = True
    FRAGMENT_NOISE_N    = 3
    HUE_NOISE_STRENGTH  = 2
    PERSPECTIVE_SKEW    = 0.05

    def __init__(self):
        self.font_paths = find_fonts()
        self.width  = 900
        self.height = 280

    def _render_word_image(self, word: str, is_english: bool = False) -> tuple:
        """رندر یک کلمه روی تصویر شفاف"""
        font_size = random.randint(38, 54)
        font = get_font(self.font_paths, font_size, is_english)

        tmp = Image.new('RGB', (1, 1))
        tmp_d = ImageDraw.Draw(tmp)
        w, h = get_text_size(tmp_d, word, font)

        pad = 20
        canvas = Image.new('RGBA', (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        base = random.randint(55, 145)
        ink = (max(20, base + random.randint(-20, 20)),
               max(20, base + random.randint(-20, 20)),
               max(20, base + random.randint(-20, 20)),
               255)

        d.text((pad, pad), word, font=font, fill=ink)

        angle = random.uniform(-18, 18)
        canvas = canvas.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

        cw, ch = canvas.size
        return canvas, cw, ch, (cw // 2, ch // 2)

    def create_captcha(self, text: str) -> Image.Image:
        print("=" * 55)
        print("🎨  کپچا پیشرفته فارسی – ضد هوش مصنوعی")
        print("=" * 55)

        words_orig = text.strip().split()
        if not words_orig:
            words_orig = [text]

        # تشخیص و reshape کردن کلمات
        words_info = []
        for w in words_orig:
            is_eng = not is_persian_word(w)
            if is_eng:
                # کلمات انگلیسی را بدون تغییر نگه دار
                shaped_word = w
            else:
                shaped_word = reshape_word(w)
            words_info.append((shaped_word, is_eng))

        # ── تنظیم عرض ──
        self.width = max(900, len(words_orig) * 160 + 150)
        self.height = 280
        img = create_background(self.width, self.height)

        # ── رندر کلمات از راست به چپ ──
        x_cursor = self.width - 30
        word_centers = []

        # برای چیدمان از راست به چپ، کلمات را برعکس می‌کنیم
        # for word, is_eng in reversed(words_info):
        for word, is_eng in words_info:

            word_img, ww, wh, center = self._render_word_image(word, is_eng)
            x_cursor -= ww

            y_pos = random.randint(
                max(10, self.height // 2 - wh // 2 - 30),
                min(self.height - wh - 10, self.height // 2 - wh // 2 + 30)
            )

            try:
                img.paste(word_img, (x_cursor, y_pos), word_img)
            except Exception:
                img.paste(word_img, (x_cursor, max(0, y_pos)))

            word_centers.append((x_cursor + center[0], y_pos + center[1]))
            x_cursor -= random.randint(8, 15)

        # ── Fake ligatures ──
        if self.FAKE_LIGATURES and len(word_centers) > 1:
            img = add_fake_ligatures(img, word_centers)

        # ── Fragment noise ──
        img = add_fragment_noise(img, self.FRAGMENT_NOISE_N)

        # ── Elastic distortion ──
        print("   • Elastic distortion ...")
        arr = np.array(img)
        arr = elastic_distort(arr, alpha=self.ELASTIC_ALPHA, sigma=self.ELASTIC_SIGMA)
        img = Image.fromarray(arr)

        # ── Perspective skew ──
        img = perspective_skew(img, self.PERSPECTIVE_SKEW)

        # ── Wave warp ──
        print("   • Wave warp ...")
        img = wave_warp(img, self.WAVE_AMP_X, self.WAVE_AMP_Y)
    
        # ── برش خودکار برای حذف حاشیه‌های سیاه ──
        img = auto_crop(img)
     
        # ── Occluding bands ──
        print("   • Occluding bands ...")
        img = add_occluding_bands(img, self.OCCLUDING_BANDS)

        # ── Pixel hue noise ──
        img = pixel_hue_noise(img, self.HUE_NOISE_STRENGTH)

        # ── نویز نقطه‌ای ──
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(200, 350)):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.point((x, y), fill=tuple(random.randint(80, 220) for _ in range(3)))

        # ── فیلتر نرم ──
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        '''
        # ── عنوان "Question" ──
        try:
            tf = get_font(self.font_paths, 18, False)
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle([5, 5, 105, 30], fill=(255, 255, 255, 190))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            ImageDraw.Draw(img).text((10, 8), "Q", font=tf, fill=(70, 70, 70))
        except Exception:
            pass
        '''
        print("=" * 55)
        print(f"✅  تصویر {self.width}×{self.height} ایجاد شد.")
        return img


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main():
    if not RESHAPER_AVAILABLE:
        print("⚠️  pip install arabic-reshaper python-bidi\n")

    text = input("📝 جمله فارسی: ").strip()
    if not text:
        text = "سلام دنیا"
        print(f"⚠️  از متن پیش‌فرض '{text}' استفاده می‌شود.")

    captcha = PersianCAPTCHAAdvanced()
    image = captcha.create_captcha(text)

    import random as _r
    fname = f"captcha_adv_{_r.randint(1000, 9999)}.png"
    image.save(fname)
    print(f"\n💾  ذخیره شد: {os.path.abspath(fname)}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 4))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()