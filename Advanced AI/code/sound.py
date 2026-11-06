import speech_recognition as sr

def recognize_speech_from_microphone():
    # ایجاد یک شناسایی کننده‌ی صوت
    recognizer = sr.Recognizer()

    # استفاده از میکروفون به عنوان ورودی
    with sr.Microphone() as source:
        print("لطفاً صحبت کنید...")
        
        # تنظیم میکروفون برای دریافت صوت واضح‌تر
        recognizer.adjust_for_ambient_noise(source)
        
        try:
            # ضبط صدا
            audio = recognizer.listen(source, timeout=10)
            
            # تبدیل صدا به متن
            text = recognizer.recognize_google(audio, language="fa-IR")  # زبان فارسی
            print("متن تشخیص داده شده: ", text)
        except sr.WaitTimeoutError:
            print("زمان برای ضبط صوت به پایان رسید.")
        except sr.UnknownValueError:
            print("متأسفم، نتوانستم چیزی تشخیص دهم.")
        except sr.RequestError as e:
            print("خطا در اتصال به سرویس گوگل: {0}".format(e))

# اجرای برنامه
recognize_speech_from_microphone()
