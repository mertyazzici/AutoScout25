# Dataset contains illegal characters, which cause problem for either translation or save as excel file.
import re

df = pd.read_csv(r"Bitirme\autoscout24_dataset_20251108.csv")


# Cleaning from illegal characters
def temizle_kontrol_karakterleri(text):
    if not isinstance(text, str):
        return ""
    # Openpyxl hatasına neden olan kontrol karakterlerini kaldırma
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = text.replace('', '')
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&#?\w+;', ' ', text)
    return text.strip()


def comprehensive_cleaner_v3(text):
    text = temizle_kontrol_karakterleri(text)  # 1. Kontrol karakterlerini temizle

    # Kalan diğer temizlik adımları (küçük harf, URL, iletişim vs.)
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # URL'leri kaldırma
    text = re.sub(r'(\d{3}[\s-]?){2}\d{4}', ' ', text)  # Telefon Numaraları
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)  # E-postaları kaldırma

    # -------------------------------------------------------------
    # GÜNCELLENMİŞ VE KAPSAMLI EMOJİ/SEMBOL TEMİZLİĞİ (V3)
    # -------------------------------------------------------------
    emoji_regex = r'['
    # Emoticonlar (Suratlar vb.)
    emoji_regex += r'\U0001F600-\U0001F64F'
    # Semboller ve Piktogramlar (El işaretleri, nesneler, hava durumu)
    emoji_regex += r'\U0001F300-\U0001F5FF'
    # Ulaşım ve Harita Sembolleri
    emoji_regex += r'\U0001F680-\U0001F6FF'
    # Bayraklar
    emoji_regex += r'\U0001F1E0-\U0001F1FF'
    # Dingbatlar (Oklar, yıldızlar, kutular) ve diğer geniş aralıklar
    emoji_regex += r'\U00002702-\U000027B0'
    emoji_regex += r'\U000024C2-\U0001F251'  # Diğer birçok sembolü kapsayan geniş aralık
    emoji_regex += r']+'

    text = re.sub(emoji_regex, ' ', text, flags=re.UNICODE)

    # Özel Semboller ve Noktalama İşaretleri (Emoji dışı kalanlar)
    # Bu, önceden tanımlı standart karakter temizliğidir:
    text = re.sub(r'[•\-—*+=#@&%!?/\\~^|]', ' ', text)

    # Tekrarlayan karakterler (AAAA -> AA)
    text = re.sub(r'(\w)\1{3,}', r'\1\1', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


# df["description"] sütununa uygulama
df["cleaned_description"] = df["description"].apply(comprehensive_cleaner_v3)



### 2.ASAMA  [] li kodlarin silinmesi
def clean_tracking_codes(text):
    """
    Autoscout24 ilan metninden belirli takip ve bayi kodlarını temizler.
    (dek:[...], [cod:...], [veicolo:...], dms:...)
    """
    if not isinstance(text, str):
        return ""

    # Metni küçük harfe çevirip işlemlere başlayalım
    text = text.lower()

    # 1. dek:[sayılar] formatını temizleme
    # Örn: dek:[10217282]
    text = re.sub(r'dek:\[\d+\]', ' ', text)

    # 2. [cod: ...] ve [veicolo: ...] formatlarını temizleme
    # Örn: [cod: 1303255 117] veya [veicolo: 371806]
    text = re.sub(r'\[(cod|veicolo):\s*.*?\]', ' ', text)

    # 3. dms: harf/sayı formatını temizleme
    # Örn: dms: u128128, dms: n 1054889, dms: u 94247
    text = re.sub(r'dms\s*:\s*[a-z0-9\s]+', ' ', text)

    # İşlem sonunda kalan fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Yeni fonksiyonu mevcut temizlenmiş sütuna uygulama ve ilerlemeyi görme

df["cleaned_description_v2"] = df["cleaned_description"].progress_apply(clean_tracking_codes)

df["cleaned_description_v2"].isnull().sum()
df = df.drop(columns=["description"])
df = df.drop(columns=["cleaned_description"])

df.rename(columns={'cleaned_description_v2': 'description'}, inplace=True)

# Yeniden adlandırdığımız description sütunu şu anda listenin sonunda (veya 1. adımda nereye koyduysanız oradadır)
yeni_description_sutunu = 'description'
# 1. 'description' sütununu listeden çıkarın
column_names.remove(yeni_description_sutunu)
# 2. İkinci sıraya (indeks 1) ekleyin
# (İndeks 0 ilk sıradır, İndeks 1 ikinci sıra demektir)
column_names.insert(1, yeni_description_sutunu)
# Sütunları bu yeni sıraya göre DataFrame'e uygulayın
df = df[column_names]

## ## extract for manual inspection if any residual remain could cause a problem
with open("Bitirme/description_dump.txt", "w", encoding="utf-8", errors="replace") as f:
    for v in df["description"]:
        f.write(str(v))
        f.write("\n")

with open("Bitirme/description_dump_repr.txt", "w", encoding="utf-8") as f:
    for v in df["description"]:
        f.write(repr(v))
        f.write("\n")

# To saving in excel format can shortcut help determine any unexpected characters remain
df.to_excel("Bitirme/auto24_description_cleaned.xlsx",index=False)

df.to_csv("Bitirme/auto24_description_cleaned.csv",index=False)






### ONCEKI DENEME ###

# def temizle_standart(text):
#     text = str(text).lower() # Tüm metni küçük harfe çevirme
#     text = re.sub(r'<[^>]+>', '', text) # HTML etiketlerini kaldırma
#     text = re.sub(r'http\S+|www\S+', '', text) # URL'leri kaldırma
#     text = re.sub(r'&#?\w+;', '', text) # HTML varlıklarını kaldırma
#     return text
#
#
# def temizle_ozel_karakter(text):
#     # Emojileri temizler (daha karmaşık bir regex gerekebilir)
#     # Basit emojiler ve özel semboller için
#     text = re.sub(r'[\U0001F600-\U0001F64F\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text, flags=re.UNICODE)
#
#     # Araç ilanlarında sıkça kullanılan özel işaretleri boşlukla değiştirme
#     text = re.sub(r'[•\-—*+=#@&%!?/\\~^|]', ' ', text)
#
#     # Birden fazla boşluğu tek boşluğa indirgeme
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     return text
#
# def tekrar_karakter_temizle(text):
#     # Dört veya daha fazla kez tekrar eden karakterleri ikiye indirir
#     text = re.sub(r'(\w)\1{3,}', r'\1\1', text)
#     return text
#
# def iletisim_temizle(text):
#     # Basit telefon numarası regex'i (farklı formatlar için geliştirilebilir)
#     text = re.sub(r'(\d{3}[\s-]?){2}\d{4}', ' ', text) # Örn: 5xx xxx xx xx
#     text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text) # E-postaları kaldırma
#     return text
#
#
# def comprehensive_cleaner(text):
#     text = temizle_standart(text)  # Küçük harf, HTML, URL
#     text = iletisim_temizle(text)  # Telefon, E-posta
#     text = temizle_ozel_karakter(text)  # Emojiler, Özel Semboller, Çoklu Boşluk
#     text = tekrar_karakter_temizle(text)  # Tekrarlayan karakterler (AAAA -> AA)
#
#     # Son bir kez fazla boşluk temizliği
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
#
# # df["description"] sütununa uygulama
# df["cleaned_description"] = df["description"].apply(comprehensive_cleaner)

# def temizle_kontrol_karakterleri(text):
#     if not isinstance(text, str):
#         return ""  # NaN veya diğer tipleri boş stringe çevir
#
#     # 1. Unicode Kontrol Karakterlerini Kaldırma (Örn: U+0000'dan U+001F'e)
#     # Openpyxl tarafından sevilmeyen çoğu gizli karakteri kaldırır
#     text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
#
#     # 2. Openpyxl'ın sorunlu gördüğü o "REPLACEMENT" karakteri (bazen  olarak görünür)
#     text = text.replace('', '')
#
#     # 3. HTML etiketlerini ve varlıklarını tekrar kontrol etme (Önceki adımdan kaldıysa)
#     text = re.sub(r'<[^>]+>', ' ', text)  # HTML etiketlerini boşlukla değiştirme
#     text = re.sub(r'&#?\w+;', ' ', text)  # HTML varlıklarını kaldırma
#
#     return text.strip()
#
#
# # Önceki temizleme fonksiyonunuzu birleştirin/güncelleyin:
# def comprehensive_cleaner_v2(text):
#     text = temizle_kontrol_karakterleri(text)  # YENİ ADIM: En başta kontrol karakterlerini temizle!
#
#     # Kalan diğer temizlik adımları (küçük harf, URL, iletişim vs.)
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+', '', text)  # URL'leri kaldırma
#     text = re.sub(r'(\d{3}[\s-]?){2}\d{4}', ' ', text)  # Telefon Numaraları
#     text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)  # E-postaları kaldırma
#
#     # Emojiler, Özel Semboller, Çoklu Boşluk
#     text = re.sub(r'[\U0001F600-\U0001F64F\U00002702-\U000027B0\U000024C2-\U0001F251]+', ' ', text, flags=re.UNICODE)
#     text = re.sub(r'[•\-—*+=#@&%!?/\\~^|]', ' ', text)
#
#     # Tekrarlayan karakterler (AAAA -> AA)
#     text = re.sub(r'(\w)\1{3,}', r'\1\1', text)
#
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
#
# # df["description"] sütununa uygulama
# df["cleaned_description"] = df["description"].apply(comprehensive_cleaner_v2)