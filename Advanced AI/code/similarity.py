# -*- coding: utf-8 -*-
"""
رنگی‌کردن جمله‌ها و پاراگراف‌های مشابه در فایل Word
نصب:
    pip install python-docx
اجرا:
    python highlight_similar_text.py

نکته: فایل اصلی دست‌نخورده می‌ماند و خروجی در فایل جداگانه ذخیره می‌شود.
"""
import os
import re
import unicodedata
import itertools
from difflib import SequenceMatcher
from tkinter import Tk, filedialog, simpledialog, messagebox

from docx import Document
from docx.shared import RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


# هر گروه از متن‌های مشابه رنگ متفاوتی می‌گیرد.
# رنگ‌ها برای پس‌زمینه متن در Word هستند.
GROUP_COLORS = [
    "C6EFCE",  # سبز روشن
    "FFC7CE",  # قرمز روشن
    "FFEB9C",  # زرد روشن
    "CFE2F3",  # آبی روشن
    "D9EAD3",  # سبز ملایم
    "F4CCCC",  # قرمز ملایم
    "D9D2E9",  # بنفش روشن
    "FCE4D6",  # نارنجی روشن
    "DDEBF7",  # آبی ملایم
    "E2F0D9",  # سبز خیلی روشن
]


def normalize(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    # یکسان‌سازی فاصله‌ها و حذف علائم نگارشی برای مقایسه
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def split_sentence_spans(text):
    """جمله‌ها را همراه با محدوده شروع/پایان در پاراگراف برمی‌گرداند."""
    spans = []
    # جداکننده بعد از نشانه‌های پایان جمله یا شکست خط
    for match in re.finditer(r".+?(?:[.!?؟؛]+(?=\s|$)|\n+|$)", text, flags=re.S):
        start, end = match.span()
        raw = match.group()
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        start += left_trim
        end = match.start() + right_trim
        sentence = text[start:end].strip()
        if len(normalize(sentence)) >= 25:
            spans.append((start, end, sentence))
    # اگر متن به علت نشانه‌گذاری نامتعارف به جمله تقسیم نشد، کل متن را یک مورد بگیر
    if not spans and len(normalize(text)) >= 25:
        spans.append((0, len(text), text.strip()))
    return spans


def set_run_shading(run, fill):
    """رنگ پس‌زمینه سفارشی برای یک run در Word."""
    r_pr = run._element.get_or_add_rPr()
    shd = r_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        r_pr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)


def iter_paragraphs(container):
    """پاراگراف‌های سند و جدول‌های تو در تو را پیمایش می‌کند."""
    for paragraph in getattr(container, "paragraphs", []):
        yield paragraph
    for table in getattr(container, "tables", []):
        for row in table.rows:
            for cell in row.cells:
                yield from iter_paragraphs(cell)


def iter_all_paragraphs(doc):
    # بدنه اصلی و جدول‌ها
    yield from iter_paragraphs(doc)


def make_clusters(items, threshold):
    """
    items: list of dicts with 'norm'
    خروجی: فهرست خوشه‌ها؛ هر خوشه شامل اندیس متن‌های مشابه است.
    """
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # حذف متن‌های یکسان نرمال‌شده از مقایسه‌های غیرضروری، ولی نگه داشتن همه محل‌ها
    for i, j in itertools.combinations(range(n), 2):
        a, b = items[i]["norm"], items[j]["norm"]
        if not a or not b:
            continue
        # پیش‌فیلتر ساده برای کاهش زمان مقایسه متن‌هایی با طول خیلی متفاوت
        if min(len(a), len(b)) / max(len(a), len(b)) < max(0.35, threshold - 0.35):
            continue
        if SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold:
            union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def rebuild_paragraph_with_highlights(paragraph, spans_and_colors):
    """
    متن پاراگراف را با runهای رنگی بازسازی می‌کند.
    این روش قالب‌بندی درون‌متنی، لینک‌ها و بعضی ویژگی‌های خاص runها را حفظ نمی‌کند؛
    سبک پاراگراف حفظ می‌شود. فقط زمانی استفاده می‌شود که جمله مشابهی پیدا شده باشد.
    """
    original = paragraph.text
    if not original:
        return

    # در صورت وجود هم‌پوشانی، اولویت با اولین محدوده است.
    spans_and_colors = sorted(spans_and_colors, key=lambda x: (x[0], x[1]))
    clean_spans = []
    last_end = 0
    for start, end, color in spans_and_colors:
        start = max(start, last_end)
        if end > start:
            clean_spans.append((start, end, color))
            last_end = end

    if not clean_spans:
        return

    # حفظ سبک پاراگراف؛ متن و قالب‌بندی character-level بازسازی می‌شود.
    paragraph.clear()
    cursor = 0
    for start, end, color in clean_spans:
        if start > cursor:
            paragraph.add_run(original[cursor:start])
        run = paragraph.add_run(original[start:end])
        set_run_shading(run, color)
        cursor = end
    if cursor < len(original):
        paragraph.add_run(original[cursor:])


def main():
    root = Tk()
    root.withdraw()

    input_path = filedialog.askopenfilename(
        title="فایل Word را انتخاب کنید",
        filetypes=[("Word document", "*.docx")]
    )
    if not input_path:
        root.destroy()
        return

    threshold_pct = simpledialog.askfloat(
        "آستانه شباهت",
        "حداقل شباهت را وارد کنید (پیشنهاد: 80):",
        initialvalue=80.0,
        minvalue=50.0,
        maxvalue=100.0,
        parent=root
    )
    if threshold_pct is None:
        root.destroy()
        return
    threshold = threshold_pct / 100.0

    try:
        doc = Document(input_path)
    except Exception as exc:
        messagebox.showerror("خطا", f"فایل باز نشد:\n{exc}", parent=root)
        root.destroy()
        return

    paragraphs = []
    for p_index, p in enumerate(iter_all_paragraphs(doc)):
        text = re.sub(r"\s+", " ", p.text).strip()
        if text:
            paragraphs.append({"index": p_index, "paragraph": p, "text": text})

    # 1) پیدا کردن پاراگراف‌های مشابه
    para_items = []
    for item in paragraphs:
        norm = normalize(item["text"])
        if len(norm) >= 35:
            para_items.append({
                "norm": norm,
                "paragraph_index": item["index"],
                "text": item["text"],
                "paragraph": item["paragraph"],
            })
    para_groups = make_clusters(para_items, threshold)

    # رنگ گروه‌های پاراگرافی؛ فقط اگر آن پاراگراف جمله مشابهی نداشته باشد،
    # در مرحله بعد ممکن است رنگ جمله‌ها نیز اعمال شود.
    para_color_by_index = {}
    for group_no, group in enumerate(para_groups):
        color = GROUP_COLORS[group_no % len(GROUP_COLORS)]
        for item_idx in group:
            para_color_by_index[para_items[item_idx]["paragraph_index"]] = color

    # 2) پیدا کردن جمله‌های مشابه در همه پاراگراف‌ها
    sentence_items = []
    for item in paragraphs:
        for start, end, sentence in split_sentence_spans(item["text"]):
            sentence_items.append({
                "norm": normalize(sentence),
                "paragraph_index": item["index"],
                "start": start,
                "end": end,
                "text": sentence,
            })

    sent_groups = make_clusters(sentence_items, threshold)

    # رنگ گروه هر جمله را ثبت کن
    sentence_color_by_item = {}
    spans_by_paragraph = {}
    for group_no, group in enumerate(sent_groups):
        color = GROUP_COLORS[group_no % len(GROUP_COLORS)]
        for item_idx in group:
            sentence_color_by_item[item_idx] = color
            it = sentence_items[item_idx]
            spans_by_paragraph.setdefault(it["paragraph_index"], []).append(
                (it["start"], it["end"], color)
            )

    # 3) اعمال رنگ‌ها به جمله‌ها؛ اگر جمله مشابه نبود ولی پاراگراف مشابه بود،
    # کل پاراگراف را با رنگ گروه پاراگرافی مشخص کن.
    paragraphs_by_index = {item["index"]: item["paragraph"] for item in paragraphs}
    for p_index, spans in spans_by_paragraph.items():
        paragraph = paragraphs_by_index.get(p_index)
        if paragraph is not None:
            # متن اصلی را پیش از بازسازی با همان متن استانداردشده بررسی می‌کنیم.
            # برای جلوگیری از اختلاف فاصله‌ها، محدوده‌ها بر مبنای متن خود پاراگراف تنظیم می‌شود.
            original = paragraph.text
            # محدوده‌ها در متن فشرده‌شده استخراج شدند؛ در اغلب اسناد معمولی برابر است.
            # اگر اختلاف داشته باشد، جمله را بر اساس یافتن متن در پاراگراف پیدا می‌کنیم.
            fixed_spans = []
            # محاسبه مجدد موقعیت جمله‌ها در متن دقیق پاراگراف
            search_from = 0
            for sentence_idx, sentence_item in enumerate(sentence_items):
                if sentence_item["paragraph_index"] != p_index:
                    continue
                matching_color = sentence_color_by_item.get(sentence_idx)
                if matching_color:
                    sentence = sentence_item["text"]
                    pos = original.find(sentence, search_from)
                    if pos < 0:
                        pos = original.find(sentence)
                    if pos >= 0:
                        fixed_spans.append((pos, pos + len(sentence), matching_color))
                        search_from = pos + len(sentence)
            if fixed_spans:
                rebuild_paragraph_with_highlights(paragraph, fixed_spans)

    # رنگ کردن پاراگراف‌های مشابهی که هنوز هیچ جمله‌ای در آن‌ها رنگ نشده است
    for item in para_items:
        p_index = item["paragraph_index"]
        if p_index in para_color_by_index and p_index not in spans_by_paragraph:
            paragraph = item["paragraph"]
            for run in paragraph.runs:
                set_run_shading(run, para_color_by_index[p_index])

    base, ext = os.path.splitext(input_path)
    output_path = base + "_متن‌های_مشابه_رنگی.docx"
    try:
        doc.save(output_path)
    except Exception as exc:
        messagebox.showerror(
            "خطا در ذخیره",
            f"فایل خروجی ذخیره نشد. اگر فایل در Word باز است آن را ببندید.\n\n{exc}",
            parent=root
        )
        root.destroy()
        return

    root.destroy()
    root2 = Tk()
    root2.withdraw()
    messagebox.showinfo(
        "پایان بررسی",
        f"فایل رنگی ذخیره شد:\n{output_path}\n\n"
        f"گروه‌های پاراگراف مشابه: {len(para_groups)}\n"
        f"گروه‌های جمله مشابه: {len(sent_groups)}\n\n"
        "هر گروه مشابه رنگ مشترک دارد؛ گروه‌های بعدی رنگ متفاوت می‌گیرند.\n"
        "متن اصلی تغییر نمی‌کند، اما بازسازی جمله‌های رنگی ممکن است قالب‌بندی داخل همان پاراگراف "
        "مانند bold/italic یا لینک را حفظ نکند."
    )
    root2.destroy()


if __name__ == "__main__":
    main()
