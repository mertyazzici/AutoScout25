# =============================================================================
# Miull Bitirme Çalışması | Grup: DropNA
# Yazarlar: GC, MKB, MO, MY, OFO
# Temize gecilmis hali
# =============================================================================

import pandas as pd
import numpy as np
import re
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.exceptions import ConvergenceWarning

# =============================================================================
# Uyarılar ve Görselleştirme Ayarları Ozellestirme
# =============================================================================

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# =============================================================================
# Opsiyonel Görünüm Ayarları (EDA sırasında gerektiğinde açılabilir)
# =============================================================================
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
# pd.set_option("display.width", 200)
# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.float_format", lambda x: "%.3f" % x)

# =============================================================================
# Yardımcı Fonksiyonlar
# =============================================================================

def value_counts_with_nan(series):
    """
    Bir değişkenin NaN dahil değer dağılımını ve NaN sayısını döndürür.
    """
    counts = series.value_counts(dropna=False)
    nan_count = series.isna().sum()
    return counts, nan_count

def check_df(dataframe, head=5):
    """
    DataFrame için hızlı genel resim kontrolü.
    """
    print("##################### SEKLI #####################")
    print(dataframe.shape)

    print("\n##################### DATA TURLERI #####################")
    print(dataframe.dtypes)

    print("\n##################### ILK SATIRLAR #####################")
    print(dataframe.head(head))

    print("\n##################### SON SATIRLAR #####################")
    print(dataframe.tail(head))

    print("\n##################### KAYIP DEGER SAYILARI #####################")
    print(dataframe.isnull().sum())

# =============================================================================
# Veri Setinin Yüklenmesi
# =============================================================================
# Datasetimizin 3 sutunu tercume isleminden gectikten sonra en ideal kullanilabilir hale gelmistir.
# Description, model_version, body_color_original farkli dillerde icerik barindiriyordu.
# En guvenli dil olarak Ingilizceye cevirip veriden maximum bilgi cikartabilecegimize karar verdik.
# (tercume icin gereken script ayri tutuldu)
df = pd.read_csv("Bitirme/auto_translated_v1.csv",low_memory=False)

# =============================================================================
# Genel Resim (İlk İnceleme) - EDA Sureci
# =============================================================================

check_df(df)

print("\nTekrarlı gözlem sayısı:", df.duplicated().sum())
print("Eksik değer var mı?:", df.isnull().values.any())

# =============================================================================
# Icerik Yazisi cikartma Harici Inceleme icin
# =============================================================================

# with open("Bitirme/description_dump.txt", "w", encoding="utf-8", errors="replace") as f:
#     for v in df["description_en"]:
#         f.write(str(v))
#         f.write("\n")
#
# with open("Bitirme/model_version_en_dump.txt", "w", encoding="utf-8", errors="replace") as f:
#     for v in df["model_version_en"]:
#         f.write(str(v))
#         f.write("\n")

# =============================================================================
# Değişken Türlerinin Belirlenmesi
# =============================================================================

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki değişkenleri;
    - kategorik
    - numerik
    - kategorik görünümlü kardinal

    olacak şekilde ayırır.

    Parameters
    ----------
    dataframe : pd.DataFrame
        İncelenecek veri seti
    cat_th : int
        Numerik olup kategorik kabul edilecek eşik değer
    car_th : int
        Kategorik olup kardinal kabul edilecek eşik değer

    Returns
    -------
    cat_cols : list
        Kategorik değişkenler
    num_cols : list
        Numerik değişkenler
    cat_but_car : list
        Kategorik görünümlü kardinal değişkenler
    """

    # Kategorik değişkenler
    cat_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype.name in ["category", "object", "bool"]
    ]

    # Numerik görünümlü fakat kategorik değişkenler
    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th
        and dataframe[col].dtype.name in ["int64", "float64"]
    ]

    # Kategorik görünümlü fakat kardinal değişkenler
    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th
        and dataframe[col].dtype.name in ["category", "object"]
    ]

    # Nihai kategorik liste
    cat_cols = list(set(cat_cols + num_but_cat) - set(cat_but_car))

    # Numerik değişkenler
    num_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype.name in ["int64", "float64"]
        and col not in cat_cols
    ]

    print(f"Gözlem Sayısı: {dataframe.shape[0]}")
    print(f"Değişken Sayısı: {dataframe.shape[1]}")
    print(f"Kategorik Değişken Sayısı: {len(cat_cols)}")
    print(f"Numerik Değişken Sayısı: {len(num_cols)}")
    print(f"Kardinal Değişken Sayısı: {len(cat_but_car)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Observations: 118382
# Variables: 71

# =====================================================================================
# cat_cols: 27
# cat_cols: ['price_currency', 'price_tax_deductible', 'price_negotiable', 'vehicle_type', 'body_type', 'body_color', 'paint_type', 'upholstery', 'upholstery_color', 'transmission', 'drive_train', 'has_particle_filter', 'fuel_category', 'primary_fuel', 'is_used', 'is_new', 'is_preregistered', 'had_accident', 'has_full_service_history', 'non_smoking', 'is_rental', 'envir_standard', 'offer_type', 'country_code', 'seller_is_dealer', 'seller_type', 'nr_doors']
# Get Dummies #####

# num_cols: 23
# num_cols: ['ratings_count', 'ratings_recommend_percentage', 'price', 'price_net', 'price_vat_rate', 'mileage_km_raw', 'production_year', 'nr_seats', 'power_kw', 'power_hp', 'gears', 'cylinders', 'cylinders_volume_cc', 'electric_range_km', 'electric_range_city_km', 'fuel_cons_comb_l100_km', 'co2_emission_grper_km', 'fuel_cons_comb_l100_wltp_km', 'fuel_cons_electric_comb_l100_wltp_km', 'co2_emission_grper_wltp_km', 'nr_prev_owners', 'latitude', 'longitude']
# Check Outliers #####

# cat_but_car: 21
# cat_but_car: ['id', 'description', 'ratings_average', 'vin', 'make', 'model', 'model_version_en', 'german_hsn_tsn', 'mileage_km', 'registration_date', 'body_color_original', 'weight_kg', 'equipment_comfort', 'equipment_entertainment', 'equipment_extra', 'equipment_safety', 'original_market', 'zip', 'city', 'street', 'seller_company_name']
# Are they useful to feed other columns? #####
# num_but_cat: 1
# num_but_cat: ['nr_doors']

# =============================================================================
# Kategorik Değişkenlerin İncelenmesi
# =============================================================================

def cat_summary(dataframe, col_name, plot=False):
    """
    Kategorik değişkenler için frekans ve oran bilgisi üretir.
    """
    summary_df = pd.DataFrame({
        "Count": dataframe[col_name].value_counts(),
        "Ratio (%)": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })

    print(summary_df)
    print("-" * 50)

    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.xticks(rotation=45)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# =============================================================================
# Numerik Değişkenlerin İncelenmesi
# =============================================================================

def num_summary(dataframe, col_name, plot=False):
    """
    Numerik değişkenler için özet istatistik üretir.
    """
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=False)

# =============================================================================
# Aykırı Değer Kontrolü (IQR Yöntemi)
# =============================================================================

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    IQR yöntemine göre alt ve üst sınırları hesaplar.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1

    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr

    return low_limit, up_limit


def has_outlier(dataframe, col_name):
    """
    İlgili değişkende aykırı değer olup olmadığını kontrol eder.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    return dataframe[
        (dataframe[col_name] < low_limit) |
        (dataframe[col_name] > up_limit)
    ].any(axis=None)


# Numerik değişkenler için genel durumun kontrolü
for col in num_cols:
    print(f"{col}: {has_outlier(df, col)}")

# =============================================================================
# Kardinal Değişkenlerin İncelenmesi
# =============================================================================

def cardinality_report(dataframe, column_list):
    """
    Verilen değişken listesi için benzersiz değer sayılarını döndürür.
    """
    return (
        dataframe[column_list]
        .nunique()
        .sort_values(ascending=False)
        .to_frame(name="unique_value_count")
    )


cardinality_df = cardinality_report(df, cat_but_car)
print(cardinality_df)


# =============================================================================
# Düşük Sayıda Unique Değere Sahip Değişkenler
# =============================================================================

def get_small_unique_values(dataframe, max_unique=10):
    """
    Belirtilen eşikten az benzersiz değere sahip değişkenleri listeler.
    """
    result = {}

    for col in dataframe.columns:
        if dataframe[col].nunique() < max_unique:
            result[col] = dataframe[col].dropna().unique().tolist()

    return result


small_unique_dict = get_small_unique_values(df, max_unique=10)

# =============================================================================
# Orta Seviyede Unique Değere Sahip Değişkenler
# =============================================================================

def get_medium_unique_values(dataframe, min_unique=10, max_unique=100):
    """
    Orta kardinaliteye sahip değişkenleri listeler.
    """
    result = {}

    for col in dataframe.columns:
        nunique = dataframe[col].nunique()
        if min_unique < nunique < max_unique:
            result[col] = dataframe[col].dropna().unique().tolist()

    return result


medium_unique_dict = get_medium_unique_values(df)


# =============================================================================
# EDA ÖZETİ
# =============================================================================
# - Veri seti heterojen ve çok sayıda metinsel alan içeriyor
# - Fiyat ve rating değişkenleri ciddi temizlik gerektiriyor
# - Model, açıklama ve versiyon alanları bilgi kurtarma için uygun


# =============================================================================
# FEATURE-LEVEL DATA CLEANING (Feature-level recovery & enrichment)
# Sütun Bazlı Temizlik ve Mantıksal Kontroller
# =============================================================================
# Kapsam:
# - Anlamsız değerler
# - Hatalı girilmiş değerler
# - Missing value işlemleri
# - İş kurallarına dayalı kontroller
# =============================================================================
#
# Bu aşamada veri seti sütun bazında ele alınmakta,
# modelleme sürecine uygun hale getirilmesi hedeflenmektedir.
# Target değişken olarak PRICE belirlenmiştir.
# Target'a yönelik analiz ve dönüşümler çalışmanın ilerleyen
# aşamalarında ele alınacaktır.
#

# =============================================================================
# ADIM 1: PRICE | Fiyat Değişkeni Temizleme  "price"
# =============================================================================

print("\n--- PRICE: Temizlik ve Doğrulama ---")

def clean_price(value):
    """
    Fiyat bilgisini string / bozuk formatlardan float'a çevirir.
    Geçersiz değerleri NaN olarak döndürür.
    """
    if pd.isna(value):
        return np.nan

    value = (
        str(value)
        .replace("€", "")
        .replace("-", "")
        .strip()
    )

    # Avrupa sayı formatı kontrolü
    if "," in value:
        value = value.replace(".", "").replace(",", ".")
    elif "." in value and len(value.split(".")[-1]) == 3:
        value = value.replace(".", "")

    try:
        return float(value)
    except ValueError:
        return np.nan


# Tip dönüşümü
df["price"] = df["price"].apply(clean_price)

# Fiyatı olmayan kayıtlar modelleme için anlamsız → drop
df.dropna(subset=["price"], inplace=True)

# -----------------------------------------------------------------------------
# Aykırı Değer Baskılama (IQR + İş Kuralı)
# -----------------------------------------------------------------------------

q_low = df["price"].quantile(0.05)
q_high = df["price"].quantile(0.95)
iqr = q_high - q_low

lower_bound = q_low - 1.5 * iqr
upper_bound = q_high + 1.5 * iqr

# İş kuralı: 250 € altı araç fiyatı mantıksız
lower_bound = max(lower_bound, 250)

# Alt sınır baskılama
df.loc[df["price"] < lower_bound, "price"] = lower_bound

# Üst sınır bilinçli olarak baskılanmıyor
# (Hyper car fiyatları gerçek olabilir)

print(
    f"Price temizlendi | Min: {df['price'].min():.2f} € "
    f"| Max: {df['price'].max():.2f} €"
)

# =============================================================================
# ADIM 2: PRICE FLAGS | Yan Özellikler   "price_negotiable", "price_tax_deductible"
# =============================================================================
print("\n--- [2/3] PRICE FLAGS (Negotiable & Tax) ---")

# Fiyat ile ilişkili boolean bilgilerin
# modelleme için sayısal formata dönüştürülmesi
price_flag_cols = ["price_negotiable", "price_tax_deductible"]

for col in price_flag_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .map({
            "true": 1, "yes": 1, "1": 1, "1.0": 1,
            "false": 0, "no": 0, "0": 0, "0.0": 0
        })
        .fillna(0)
        .astype(int)
    )

print("Price flag kolonları 1/0 formatına dönüştürüldü.")

# =============================================================================
# ADIM 3: RATINGS | Puanlama Değişkenleri "ratings_average", "ratings_recommend_percentage"
# =============================================================================
print("\n--- [3/3] RATINGS İŞLEMLERİ ---")

rating_cols = ["ratings_average", "ratings_recommend_percentage"]

for col in rating_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Puan bilgisi mevcut mu? (0 = yok, 1 = var)
df["NEW_IS_RATED"] = np.where(
    (df["ratings_average"].isna()) | (df["ratings_average"] == 0),
    0,
    1
)

# 0 değerleri eksik kabul edilir
df.loc[df["ratings_average"] == 0, "ratings_average"] = np.nan

# Öncelik: model bazlı medyan ile doldurma
df["ratings_average"] = df["ratings_average"].fillna(
    df.groupby("model")["ratings_average"].transform("median")
)

# Hâlâ eksik kalanlar olabilir (çok nadir modeller)
remaining_missing = df["ratings_average"].isna().sum()
print(f"Ratings Average için kalan eksik gözlem sayısı: {remaining_missing}")


# =============================================================================
# ADIM 4: MODEL | Eksik Değerlerin Metin Alanlarından Doldurulması
# =============================================================================

# Mevcut model dağılımı (kontrol amaçlı)
df["model"].value_counts()

# Eksik model maskesi
mask_missing = df["model"].isna()

# Modelin hangi kaynaktan doldurulduğunu takip etmek için
df["model_fill_source"] = np.nan

# -----------------------------------------------------------------------------
# Model sözlüğünün hazırlanması
# -----------------------------------------------------------------------------

# Mevcut model değerlerinden benzersiz liste oluştur
model_list = (
    df["model"]
    .dropna()
    .unique()
    .tolist()
)

# Uzun model isimleri önce gelecek şekilde sırala
# (örn: "Series 3" → "3" çakışmasını önlemek için)
model_list = sorted(model_list, key=len, reverse=True)

# Regex-safe pattern sözlüğü
model_patterns = {
    model: re.compile(
        rf"\b{re.escape(model)}\b",
        flags=re.IGNORECASE
    )
    for model in model_list
}

# -----------------------------------------------------------------------------
# Metin içerisinden tekil model eşleşmesi yakalama fonksiyonu
# -----------------------------------------------------------------------------

def match_single_model(text):
    """
    Verilen metin içerisinde yalnızca tek bir model eşleşmesi varsa
    ilgili modeli döndürür. Birden fazla veya hiç eşleşme yoksa None döner.
    """
    if pd.isna(text):
        return None

    matches = [
        model
        for model, pattern in model_patterns.items()
        if pattern.search(str(text))
    ]

    return matches[0] if len(matches) == 1 else None


# -----------------------------------------------------------------------------
# Model bilgisinin alternatif kolonlardan doldurulması
# -----------------------------------------------------------------------------

source_cols = ["model_version_en", "description_en"]

for col in source_cols:
    idx = mask_missing & df[col].notna()

    matched_models = df.loc[idx, col].apply(match_single_model)
    matched_models = matched_models.dropna()

    df.loc[matched_models.index, "model"] = matched_models
    df.loc[matched_models.index, "model_fill_source"] = f"dict_{col}"


# Kalan eksik model sayısı
remaining_missing_models = df["model"].isna().sum()
print(f"Model bilgisi doldurulamayan gözlem sayısı: {remaining_missing_models}")


# Manuel inceleme gerekirse:
# model_df = df[df["model"].isna()]
# model_df.to_excel("Bitirme/model_inspection.xlsx")

# =============================================================================
# ADIM 5: MODEL VERSION & HSN-TSN | Model Kurtarma (Referans Kolonlar)
# =============================================================================
# model_version_en:
# - Referans amaçlı kullanılır
# - Model bilgisini kurtarmak için faydalıdır
# - Modelleme aşamasında drop edilecektir
#
# german_hsn_tsn:
# - Almanya araç tip kodu (HSN + TSN)
# - HSN: Üretici kodu
# - TSN: Model / tip varyantı
# - Birlikte aracın net versiyonunu tanımlar
# =============================================================================


# -----------------------------------------------------------------------------
# HSN-TSN → Model Mapping (Sadece Güvenli Eşleşmeler)
# -----------------------------------------------------------------------------

# Model ve HSN-TSN bilgisi dolu olanlardan mapping çıkar
hsn_model_map = (
    df.loc[
        df["model"].notna() & df["german_hsn_tsn"].notna(),
        ["german_hsn_tsn", "model"]
    ]
    .drop_duplicates()
)

# Her HSN-TSN kaç farklı modele gidiyor?
hsn_model_unique = (
    hsn_model_map
    .groupby("german_hsn_tsn")["model"]
    .nunique()
)

# Sadece TEK bir modele giden HSN-TSN'ler güvenli kabul edilir
valid_hsn = hsn_model_unique[hsn_model_unique == 1].index

# Güvenli HSN → Model mapping
hsn_to_model = (
    hsn_model_map
    .loc[hsn_model_map["german_hsn_tsn"].isin(valid_hsn)]
    .drop_duplicates(subset="german_hsn_tsn")
    .set_index("german_hsn_tsn")["model"]
)

# -----------------------------------------------------------------------------
# Eksik Model Bilgisinin HSN-TSN Üzerinden Doldurulması
# -----------------------------------------------------------------------------

idx = (
    df["model"].isna() &
    df["german_hsn_tsn"].notna() &
    df["german_hsn_tsn"].isin(hsn_to_model.index)
)

df.loc[idx, "model"] = df.loc[idx, "german_hsn_tsn"].map(hsn_to_model)
df.loc[idx, "model_fill_source"] = "hsn_tsn"

print(f"HSN-TSN üzerinden doldurulan model sayısı: {idx.sum()}")

# -----------------------------------------------------------------------------
# Bilinçli Olarak Doldurulmayan (Ambiguous) HSN-TSN'ler
# -----------------------------------------------------------------------------

ambiguous_hsn = hsn_model_unique[hsn_model_unique > 1]
print(f"Birden fazla modele giden HSN-TSN sayısı: {ambiguous_hsn.shape[0]}")


# =============================================================================
# ADIM 6: mileage_km_raw
# Sütun Bazlı Temizlik, Kurtarma ve Mantıksal Etiketleme
# =============================================================================

# Genel görünüm
df["mileage_km_raw"].sort_values(ascending=True)
df["mileage_km_raw"].value_counts()

# -----------------------------------------------------------------------------
# Durum / Karar Özeti
# -----------------------------------------------------------------------------
# 1- Yeni araç + km missing                     → km = 0 (güvenli)
# 2- Kullanılmış + km missing + servis yok      → km = 0 (yüksek risk)
# 3- Used ama km = 0                            → çelişki
# 4- Full service history + km = 0              → mantıksal çelişki
# 5- Kurallarla doldurulamayanlar               → unresolved
# 6- Açıkça hatalı birkaç kayıt                 → manuel drop
# -----------------------------------------------------------------------------

# Tüm satırlar için başlangıç etiketi
df["mileage_fill_reason"] = "unknown"

# -----------------------------------------------------------------------------
# 1 Yeni araç → km = 0 (güvenli kurtarma)
# -----------------------------------------------------------------------------
idx_new = df["mileage_km_raw"].isna() & (df["is_new"] == True)

df.loc[idx_new, "mileage_km_raw"] = 0
df.loc[idx_new, "mileage_fill_reason"] = "new_vehicle_zero"

# -----------------------------------------------------------------------------
# 2 Kullanılmış + km missing + servis yok → riskli varsayım
# -----------------------------------------------------------------------------
idx_high_risk = (
    df["mileage_km_raw"].isna() &
    (df["is_new"] == False) &
    (df["nr_prev_owners"].isna()) &
    (df["has_full_service_history"] == False)
)

df.loc[idx_high_risk, "mileage_km_raw"] = 0
df.loc[idx_high_risk, "mileage_fill_reason"] = "assumed_zero_high_risk"

# -----------------------------------------------------------------------------
# 3 Used ama km = 0 → ilan çelişkisi
# -----------------------------------------------------------------------------
idx_used_zero = (df["mileage_km_raw"] == 0) & (df["is_used"] == True)

df.loc[idx_used_zero, "mileage_fill_reason"] = "zero_used_contradiction"

# -----------------------------------------------------------------------------
# 4 Full service history + km = 0 → mantıksal çelişki
# -----------------------------------------------------------------------------
idx_service_conflict = (
    (df["mileage_km_raw"] == 0) &
    (df["has_full_service_history"] == True)
)

df.loc[idx_service_conflict, "mileage_fill_reason"] = "zero_service_history_conflict"

# -----------------------------------------------------------------------------
# 5 Hâlâ NaN kalanlar → çözülemeyen
# -----------------------------------------------------------------------------
idx_unresolved = df["mileage_km_raw"].isna()
df.loc[idx_unresolved, "mileage_fill_reason"] = "missing_unresolved"

# -----------------------------------------------------------------------------
# 6 Manual incelemede açıkça hatalı bulunan kayıtlar
# -----------------------------------------------------------------------------
hatali_indices = [15254, 59264, 113451]
df = df.drop(index=hatali_indices).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Kontrol
# -----------------------------------------------------------------------------
df["mileage_fill_reason"].value_counts()


# =============================================================================
# ADIM 7: registration_date & production_year
# Üretim Yılı Türetme ve Kaynak Etiketleme
# =============================================================================

# -----------------------------------------------------------------------------
# Amaç:
# - registration_date, model_version ve kullanım ipuçlarından
#   production_year türetmek
# - Her atamanın kaynağını izlenebilir hale getirmek
# -----------------------------------------------------------------------------

# Tarih dönüşümü
df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")

# Başlangıç
df["production_year"] = np.nan
df["production_year_source"] = "unknown"

# =============================================================================
# description_en üzerinden production_year çıkarımı (güvenli brand new pattern)
# =============================================================================
idx_desc = df["production_year"].isna() & df["description_en"].notna()

def extract_year_from_desc(text):
    text_str = str(text).lower()

    # 1. model year pattern
    match_year = re.search(r"model year:\s*(\d{4})", text_str)
    if match_year:
        year = int(match_year.group(1))
        if 1900 <= year <= 2050:
            return year

    # 2. brand new * car pattern → production_year = 2025
    match_brand_new = re.search(r"brand new(\s\w+)?\s+car", text_str)
    if match_brand_new:
        return 2025

    return np.nan

df.loc[idx_desc, "production_year"] = df.loc[idx_desc, "description_en"].apply(extract_year_from_desc)
df.loc[idx_desc & df["production_year"].notna(), "production_year_source"] = "description_en_pattern"

# -----------------------------------------------------------------------------
# 1 Registration date → yıl (proxy, en güvenli kaynak)
# -----------------------------------------------------------------------------
idx_reg = df["registration_date"].notna()

df.loc[idx_reg, "production_year"] = df.loc[idx_reg, "registration_date"].dt.year
df.loc[idx_reg, "production_year_source"] = "registration_date"

# -----------------------------------------------------------------------------
# 2 Model version string → açık model yılı (örn: 2025, ModelYear25)
# -----------------------------------------------------------------------------
idx_model_2025 = (
    df["model_version_en"].str.contains(r"\b2025\b|ModelYear25", case=False, na=False)
)

df.loc[idx_model_2025, "production_year"] = 2025
df.loc[idx_model_2025, "production_year_source"] = "model_version"

# -----------------------------------------------------------------------------
# 3 Düşük kilometre → yeni model varsayımı (agresif ama bilinçli)
# -----------------------------------------------------------------------------
idx_low_mileage = (
    df["production_year"].isna() &
    df["mileage_km_raw"].between(1, 1000)
)

df.loc[idx_low_mileage, "production_year"] = 2025
df.loc[idx_low_mileage, "production_year_source"] = "low_mileage_assumption"

# -----------------------------------------------------------------------------
# 4 mileage_km string içinde km geçenler → yeni araç varsayımı
# -----------------------------------------------------------------------------
idx_mileage_str = (
    df["production_year"].isna() &
    df["mileage_km"].astype(str).str.contains("km", case=False, na=False)
)

df.loc[idx_mileage_str, "production_year"] = 2025
df.loc[idx_mileage_str, "production_year_source"] = "mileage_string_assumption"

# -----------------------------------------------------------------------------
# 5 Hâlâ boş kalanlar → unresolved
# -----------------------------------------------------------------------------
idx_unresolved = df["production_year"].isna()
df.loc[idx_unresolved, "production_year_source"] = "unresolved"

# Kontrol
df["production_year"].isna().sum()
df["production_year_source"].value_counts()


# =============================================================================
# vehicle_type
# DROP (heavy duty industrial)
# - Binek araç fiyat dinamiklerinden tamamen farklı
# - mileage, price ve age ilişkisini bozuyor
# - Bu çalışma kapsamı dışında bırakıldı
# =============================================================================



# =============================================================================
# ADIM 8: body_type
# Araç gövde tipi standardizasyonu ve tutarlılık kurtarma
# =============================================================================

# ------------------------------------------------------------------
# 1 Regex bazlı güçlü sinyaller (önce spesifik → sonra genel)
# ------------------------------------------------------------------

# Convertible
df.loc[
    df["model_version_en"].str.contains(r"\b(cabrio|cabriolet)\b", case=False, na=False),
    "body_type"
] = "Convertible"

# Coupe (Gran Coupe hariç tutulur)
df.loc[
    df["model_version_en"].str.contains(r"\b(coupé|coupe)\b", case=False, na=False) &
    ~df["model_version_en"].str.contains(r"gran\s+coupe", case=False, na=False),
    "body_type"
] = "Coupe"

# Performance Coupe sinyalleri (Porsche özel)
df.loc[
    df["model_version_en"].str.contains(r"\b(carrera|911|992|997|gt3)\b", case=False, na=False),
    "body_type"
] = "Coupe"

# Station Wagon
df.loc[
    df["model_version_en"].str.contains(
        r"\b(touring|avant|shooting brake|tourismo)\b",
        case=False,
        na=False
    ),
    "body_type"
] = "Station wagon"

# Sedan
df.loc[
    df["model_version_en"].str.contains(r"\b(limousine)\b", case=False, na=False),
    "body_type"
] = "Sedan"


# ------------------------------------------------------------------
# 2 Yapısal ipucu: koltuk sayısı (overwrite YOK)
# Sadece body_type boş veya Other ise
# ------------------------------------------------------------------

mask_unknown = df["body_type"].isna() | (df["body_type"] == "Other")

df.loc[
    mask_unknown & (df["nr_seats"].isin([4, 5])),
    "body_type"
] = df.loc[
    mask_unknown & (df["nr_seats"].isin([4, 5])),
    "body_type"
].fillna("Sedan")


# ------------------------------------------------------------------
# 3 make + model bazlı MODE fallback
# Sadece Other olan kayıtlar için
# ------------------------------------------------------------------

mask_other = df["body_type"] == "Other"

mode_body_type = (
    df.loc[~mask_other]
      .groupby(["make", "model"])["body_type"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

df.loc[mask_other, "body_type"] = (
    df.loc[mask_other]
      .set_index(["make", "model"])
      .index
      .map(mode_body_type)
)


# ------------------------------------------------------------------
# 4 Marka / model bazlı son güvenli override’lar
# ------------------------------------------------------------------

df.loc[(df["body_type"].isna()) & (df["make"] == "Rolls-Royce"), "body_type"] = "Sedan"
df.loc[(df["body_type"].isna()) & (df["make"] == "Mercedes-Benz"), "body_type"] = "Sedan"
df.loc[(df["body_type"].isna()) & (df["model"] == "Transit Connect"), "body_type"] = "Van"
df.loc[(df["body_type"].isna()) & (df["make"] == "Honda"), "body_type"] = "Sedan"


# ------------------------------------------------------------------
# 5 Kontrol
# ------------------------------------------------------------------

df["body_type"].isnull().sum()



# =============================================================================
# ADIM 9: nr_seats
# Koltuk sayısı standardizasyonu ve çok katmanlı kurtarma
# =============================================================================

# ------------------------------------------------------------------
# 0️⃣ İlk kontrol
# ------------------------------------------------------------------
df["nr_seats"].isnull().sum()


# ------------------------------------------------------------------
# 1️⃣ Katman: Açıkça hatalı / encode edilmiş değerlerin düzeltilmesi
# (Bitmask / yanlış encoding kaynaklı)
# ------------------------------------------------------------------

seat_fix_map = {
    255: 7.0,
    127: 5.0,
    55: 5.0,
    15: 5.0,
    14: 5.0
}

df["nr_seats"] = df["nr_seats"].replace(seat_fix_map)


# ------------------------------------------------------------------
# 2️⃣ Katman: nr_doors + model / marka sinyalleri
# (nr_seats == 1 genellikle hatalı)
# ------------------------------------------------------------------

df.loc[(df["nr_seats"] == 1) & (df["nr_doors"] == 5), "nr_seats"] = 5.0
df.loc[(df["nr_seats"] == 1) & (df["nr_doors"] == 4), "nr_seats"] = 4.0

# Spor / roadster modeller
df.loc[
    (df["nr_seats"] == 1) &
    (df["model"].str.contains(r"vantage|cayman|z4", case=False, na=False)),
    "nr_seats"
] = 2.0

# Marka / model bazlı güvenli override’lar
df.loc[(df["nr_seats"] == 1) & (df["make"] == "Volvo"), "nr_seats"] = 5.0
df.loc[(df["nr_seats"] == 1) & (df["model"].isin(["X4", "X5"])), "nr_seats"] = 5.0
df.loc[(df["nr_seats"] == 1) & (df["make"] == "Porsche"), "nr_seats"] = 4.0
df.loc[(df["nr_seats"] == 1) & (df["model"] == "128"), "nr_seats"] = 5.0
df.loc[(df["nr_seats"] == 1) & (df["model"] == "235"), "nr_seats"] = 2.0


# ------------------------------------------------------------------
# 3 Katman: body_type + kapı sayısı sinyali
# ------------------------------------------------------------------

df.loc[
    df["nr_seats"].isna() &
    (df["body_type"] == "Station wagon") &
    (df["nr_doors"] == 5),
    "nr_seats"
] = 5.0

df.loc[
    df["nr_seats"].isna() &
    (df["body_type"] == "Off-Road/Pick-up") &
    (df["nr_doors"] == 5),
    "nr_seats"
] = 5.0


# ------------------------------------------------------------------
# 4 Katman: make + model MODE fallback
# ------------------------------------------------------------------

mode_seats_mm = (
    df.groupby(["make", "model"])["nr_seats"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

df.loc[df["nr_seats"].isna(), "nr_seats"] = (
    df.loc[df["nr_seats"].isna()]
      .set_index(["make", "model"])
      .index
      .map(mode_seats_mm)
)


# ------------------------------------------------------------------
# 5 Katman: model_version_en → model_key MODE fallback
# (Son çare, daha zayıf sinyal)
# ------------------------------------------------------------------

df["model_key"] = (
    df["model_version_en"]
      .astype(str)
      .str.lower()
      .str.split()
      .str[0]
)

mode_seats_key = (
    df.groupby("model_key")["nr_seats"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

df.loc[df["nr_seats"].isna(), "nr_seats"] = (
    df.loc[df["nr_seats"].isna(), "model_key"]
      .map(mode_seats_key)
)

# ------------------------------------------------------------------
# 6 Manual fix (inceleme ile doğrulanmış edge case’ler)
# ------------------------------------------------------------------

df.loc[89031, "nr_seats"] = 5.0
df.loc[95418, "nr_seats"] = 4.0


# ------------------------------------------------------------------
# 7 Final kontrol
# ------------------------------------------------------------------

df["nr_seats"].isnull().sum()
# ~78 adet kalıyor → 2. review / manual checkup için uygun


# =============================================================================
# ADIM 10: nr_doors
# Kapı sayısı standardizasyonu ve çok katmanlı kurtarma
# =============================================================================

# ------------------------------------------------------------------
# 0 İlk kontrol
# ------------------------------------------------------------------
df["nr_doors"].isnull().sum()

# ------------------------------------------------------------------
# 1 Katman: Açıkça hatalı / encode edilmiş değerlerin düzeltilmesi
# ------------------------------------------------------------------

# Bitmask / yanlış encode
df.loc[df["nr_doors"] == 9, "nr_doors"] = 5.0

# ------------------------------------------------------------------
# 2 Katman: make + model MODE fallback
# (en güçlü sinyal)
# ------------------------------------------------------------------

mode_doors_mm = (
    df.groupby(["make", "model"])["nr_doors"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

df.loc[df["nr_doors"].isna(), "nr_doors"] = (
    df.loc[df["nr_doors"].isna()]
      .set_index(["make", "model"])
      .index
      .map(mode_doors_mm)
)

# ------------------------------------------------------------------
# 3 Katman: body_type + koltuk sayısı sinyali
# ------------------------------------------------------------------

df.loc[
    df["body_type"].str.contains("breakdown truck|car transport|box", case=False, na=False),
    "nr_doors"
] = 2.0

df.loc[
    df["nr_doors"].isna() &
    (df["body_type"] == "Off-Road/Pick-up") &
    (df["nr_seats"] == 5),
    "nr_doors"
] = 5.0

df.loc[
    df["nr_doors"].isna() &
    (df["body_type"] == "Station wagon") &
    (df["nr_seats"] == 5),
    "nr_doors"
] = 5.0

# ------------------------------------------------------------------
# 4 Katman: model_version_en → model_key MODE fallback
# (daha zayıf sinyal, son çare)
# ------------------------------------------------------------------

df["model_key"] = (
    df["model_version_en"]
      .astype(str)
      .str.lower()
      .str.split()
      .str[0]
)

mode_doors_key = (
    df.groupby("model_key")["nr_doors"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

df.loc[df["nr_doors"].isna(), "nr_doors"] = (
    df.loc[df["nr_doors"].isna(), "model_key"]
      .map(mode_doors_key)
)

# ------------------------------------------------------------------
# 5 Katman: Marka bazlı güvenli manuel override’lar
# ------------------------------------------------------------------

df.loc[df["nr_doors"].isna() & (df["make"] == "Rolls-Royce"), "nr_doors"] = 4.0
df.loc[df["nr_doors"].isna() & (df["make"] == "Bugatti"), "nr_doors"] = 2.0

# ------------------------------------------------------------------
# 6 Final kontrol
# ------------------------------------------------------------------

df["nr_doors"].isnull().sum()


# =============================================================================
# ADIM 11: body_color
# Renk bilgisi kurtarma (regex tabanlı, çok kaynaklı)
# =============================================================================

# ------------------------------------------------------------------
# 0 Ön kontrol
# ------------------------------------------------------------------
df["body_color"].isnull().sum()
df[(df["body_color"].isna()) & (df["body_color_original_en"].notna())].index.size


# ------------------------------------------------------------------
# 1 Güvenli renk listesi (tek kelimelik, gerçek renkler)
# ------------------------------------------------------------------
colors_list = [
    "beige", "black", "blue", "brown",
    "gold", "gray", "grey", "green",
    "orange", "purple", "red", "silver",
    "white", "yellow", "turquoise",
    "maroon", "burgundy", "champagne"
]

# Kelime sınırı + case insensitive
pattern = r"(?i)\b(" + "|".join(colors_list) + r")\b"


# ------------------------------------------------------------------
# 2 Renk çıkarma fonksiyonu
# ------------------------------------------------------------------
def extract_color(text):
    """
    Metin içinde tanımlı renklerden ilk güvenli eşleşmeyi döndürür.
    Birden fazla renk geçse bile yalnızca ilk match alınır.
    """
    if pd.isna(text):
        return None

    match = re.search(pattern, str(text))
    if match:
        return match.group(0).capitalize()

    return None


# ------------------------------------------------------------------
# 3 Üç farklı kaynaktan renk arama (öncelik sırasıyla)
# ------------------------------------------------------------------

# En güvenilir: orijinal renk alanı
found_in_original = df["body_color_original_en"].apply(extract_color)

# Model versiyonu (örn: "BMW 320i Black")
found_in_model = df["model_version_en"].apply(extract_color)

# Açıklama metni (en zayıf sinyal)
found_in_desc = df["description_en"].apply(extract_color)


# ------------------------------------------------------------------
# 4 Öncelik sırasına göre birleştirme
# original → model → description
# ------------------------------------------------------------------
merged_colors = (
    found_in_original
        .combine_first(found_in_model)
        .combine_first(found_in_desc)
)

# ------------------------------------------------------------------
# 5 Sadece eksik olan body_color alanlarını doldur
# ------------------------------------------------------------------
df["body_color"] = df["body_color"].fillna(merged_colors)


# ------------------------------------------------------------------
# 6 Hâlâ boş kalanlar → Other
# (ML stabilitesi için)
# ------------------------------------------------------------------
df["body_color"] = df["body_color"].fillna("Other")


# ------------------------------------------------------------------
# 7 Final kontrol
# ------------------------------------------------------------------
df["body_color"].value_counts()
df["body_color"].isnull().sum()



# =============================================================================
# ADIM 12: paint_type
# Metallic boya bilgisi kurtarma
# =============================================================================

# ------------------------------------------------------------------
# 1) description_en içinde "metallic" geçenler (yüksek güven)
# ------------------------------------------------------------------
desc_metallic = df["description_en"].str.contains(
    r"(?i)\bmetallic\b", na=False
)

# ------------------------------------------------------------------
# 2) body_color_original_en içinde "met" geçenler
# (bu sütun renk odaklı olduğu için kabul edilebilir)
# ------------------------------------------------------------------
color_met = df["body_color_original_en"].str.contains(
    r"(?i)met", na=False
)

# ------------------------------------------------------------------
# 3) paint_type sadece BOŞ olanlar doldurulur
# ------------------------------------------------------------------
paint_type_missing = df["paint_type"].isna()

df.loc[
    paint_type_missing & (desc_metallic | color_met),
    "paint_type"
] = "Metallic"


# ------------------------------------------------------------------
# 4) Son normalize & fallback
# ------------------------------------------------------------------
df["paint_type"] = (
    df["paint_type"]
        .astype("string")
        .fillna("Others")
)

# ------------------------------------------------------------------
# 5) Final kontrol
# ------------------------------------------------------------------
df["paint_type"].value_counts(dropna=False)


# =============================================================================
# ADIM 13: upholstery
# Döşeme tipi kurtarma (Text + Equipment, overwrite-safe, case-safe)
# =============================================================================

# ---------------------------------------------------------
# 0. Mevcut dağılım
# ---------------------------------------------------------
df["upholstery"].value_counts(dropna=False)

# ---------------------------------------------------------
# 0.1 Dictionary düzeltme / Normalize (case-safe)
# ---------------------------------------------------------
upholstery_fix_map = {
    "alcantara": "Alcantara",
    "Leather": "Part leather",
    "Others": "Other"
}

df["upholstery"] = df["upholstery"].replace(upholstery_fix_map)

# ---------------------------------------------------------
# 1. TEXT KATMANI – model_version_en + description_en üzerinden
# ---------------------------------------------------------
upholstery_text_map = {
    "Alcantara": [r"\balcantara\b", r"\bamaretta\b", r"\bsuede\b"],
    "Full leather": [
        r"\bfull leather\b", r"\bleather interior\b", r"\bvollleder\b",
        r"\bcuir integral\b", r"\bpelle completa\b", r"\btout cuir\b",
        r"\blederausstattung\b"
    ],
    "Part leather": [
        r"\bpart leather\b", r"\bhalf leather\b", r"\bsemi[-\s]?leather\b",
        r"\bteilleder\b", r"\bdemi[-\s]?cuir\b", r"\bpelle parziale\b",
        r"\bpartial leather\b"
    ],
    "Velour": [r"\bvelour\b", r"\bvelours\b", r"\bvelluto\b"],
    "Cloth": [
        r"\bcloth\b", r"\bfabric\b", r"\bstoff\b", r"\btissu\b",
        r"\btessuto\b", r"\btextile\b", r"\btekstil\b"
    ]
}

def extract_upholstery_text(row):
    # overwrite-safe: mevcut değer varsa dokunma
    if pd.notna(row.get("upholstery")) and str(row.get("upholstery")).strip() != "":
        return row["upholstery"]

    text = ""
    if pd.notna(row.get("model_version_en")):
        text += " " + str(row["model_version_en"])
    if pd.notna(row.get("description_en")):
        text += " " + str(row["description_en"])

    if text.strip() == "":
        return np.nan

    for category, patterns in upholstery_text_map.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return category  # Hedef kelime Title Case
    return np.nan

df["upholstery"] = df.apply(extract_upholstery_text, axis=1)

# ---------------------------------------------------------
# 2. EQUIPMENT KATMANI – sadece mantıklı anahtar kelimeler
# ---------------------------------------------------------
equip_cols = [
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety"
]

equipment_upholstery_map = {
    "Alcantara": ["alcantara"],
    "Full leather": ["leather seats", "full leather seats"],
    "Part leather": ["half leather seats", "part leather seats"]
}

def extract_upholstery_equipment(row):
    # overwrite-safe
    if pd.notna(row.get("upholstery")) and str(row.get("upholstery")).strip() != "":
        return row["upholstery"]

    for col in equip_cols:
        items = row.get(col)
        if isinstance(items, list):
            for item in items:
                item_lower = str(item).lower()
                for category, keywords in equipment_upholstery_map.items():
                    for kw in keywords:
                        if kw in item_lower:
                            return category  # Hedef kelime Title Case
    return np.nan

df["upholstery"] = df.apply(extract_upholstery_equipment, axis=1)

# ---------------------------------------------------------
# 3. FINAL NORMALIZE & FALLBACK
# ---------------------------------------------------------
# Hedef kelimeler Title Case, boş olanlara "Other"
df["upholstery"] = df["upholstery"].fillna("Other")

# ---------------------------------------------------------
# 4. FINAL KONTROL
# ---------------------------------------------------------
df["upholstery"].value_counts(dropna=False)


# =============================================================================
# =============================================================================
# upholstery_color
#rengin modeli etkileyeceğini düşünmüyoruz drop edilebilir
# =============================================================================
# =============================================================================


# =============================================================================
# ADIM 14: power_kw
# kW kolonunu temizleme, outlier düzeltme ve missing value fill
# =============================================================================

# 1. power_hp ile çakışan sütunu drop
df = df.drop(columns=["power_hp"], errors='ignore')

# 2. numeric dönüşüm
df['power_kw'] = pd.to_numeric(df['power_kw'], errors='coerce')

# 3. mantıksal sınırlar ile outlier temizleme
df.loc[df['power_kw'] < 20, 'power_kw'] = np.nan
df.loc[df['power_kw'] > 1103, 'power_kw'] = np.nan

# 4. HP olarak girilmiş olabilecek hatalı değerleri düzeltme
def correct_hp_values(power_series):
    median_val = power_series.median()

    return power_series.apply(
        lambda x: median_val if (pd.notnull(x) and x > 1000) else x
    )

df['power_kw'] = (df.groupby(['make', 'model', 'model_version_en'])['power_kw'].transform(correct_hp_values))

# 5. Missing value filling kademeli
# 1. level: make + model + model_version_en
df['power_kw'] = df.groupby(['make','model','model_version_en'])['power_kw'].transform(
    lambda x: x.fillna(x.median())
)

# 2. level: make + model
df['power_kw'] = df.groupby(['make','model'])['power_kw'].transform(
    lambda x: x.fillna(x.median())
)

# 3. level: model
df['power_kw'] = df.groupby(['model'])['power_kw'].transform(
    lambda x: x.fillna(x.median())
)

# 4. level: make
df['power_kw'] = df.groupby(['make'])['power_kw'].transform(
    lambda x: x.fillna(x.median())
)

# 6. Son kontrol
print(df['power_kw'].describe())
print("Null count:", df['power_kw'].isnull().sum())

# Manual inceleme
# kwinceleme=df.groupby(['make','model','model_version_en'])['power_kw'].describe()
# kwinceleme.to_excel("Bitirme/kwinceleme.xlsx")


# =============================================================================
# ADIM 15: transmission
# Transmission filling – multi-layer, overwrite-safe
# =============================================================================

# ---------------------------------------------------------
# 0. Mevcut dağılım
# ---------------------------------------------------------
df["transmission"].value_counts(dropna=False)

# ---------------------------------------------------------
# 1. KATMAN – Make / Model / Model Version medyan mod filling
# ---------------------------------------------------------
ref_cols = ['make', 'model', 'model_version_en']

mode_map = (
    df.dropna(subset=['transmission'])
      .groupby(ref_cols)['transmission']
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
)

df['transmission'] = df['transmission'].fillna(
    df[ref_cols].apply(lambda r: mode_map.get(tuple(r)), axis=1)
)

# ---------------------------------------------------------
# 2. KATMAN – Make / Model / Fuel Category / Power Bin
# ---------------------------------------------------------
df['power_kw_bin'] = pd.cut(
    df['power_kw'],
    bins=[0, 75, 120, 180, 250, 400, 1000],
    labels=['low', 'mid', 'upper', 'high', 'sport', 'hyper']
)

ref_cols_l2 = ['make', 'model', 'fuel_category', 'power_kw_bin']

mode_map_l2 = (
    df.dropna(subset=['transmission'])
      .groupby(ref_cols_l2)['transmission']
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
)

mask = df['transmission'].isna()
df.loc[mask, 'transmission'] = (
    df.loc[mask, ref_cols_l2]
      .apply(lambda r: mode_map_l2.get(tuple(r)), axis=1)
)

# ---------------------------------------------------------
# 3. KATMAN – Electric / Hybrid / Gear based rules
# ---------------------------------------------------------
mask = df['transmission'].isna()

# Electric & Hybrid → Automatic
df.loc[
    mask & df['fuel_category'].str.contains('electric|hybrid', case=False, na=False),
    'transmission'
] = 'Automatic'

# Çok vitesli → Automatic
df.loc[
    mask & (df['gears'] >= 7),
    'transmission'
] = 'Automatic'

# Düşük güç + az vites → Manual
df.loc[
    mask & (df['gears'] <= 5) & (df['power_kw'] <= 85),
    'transmission'
] = 'Manual'

# ---------------------------------------------------------
# 4. KATMAN – Text-based inference (description + model_version)
# ---------------------------------------------------------
def infer_transmission_from_text(text):
    if pd.isna(text):
        return None

    text = text.lower()

    auto_patterns = [
        r'\bautomatic\b', r'\bauto\b', r'\bat\b',
        r'\bdsg\b', r'\btiptronic\b',
        r'\bautomatik\b'
    ]

    manual_patterns = [
        r'\bmanual\b', r'\bmanuell\b',
        r'\b6-speed\b', r'\b5-speed\b',
        r'\bschaltgetriebe\b'
    ]

    semi_patterns = [
        r'\bsemi-automatic\b',
        r'\bautomated manual\b'
    ]

    if any(re.search(p, text) for p in semi_patterns):
        return 'Semi-automatic'
    if any(re.search(p, text) for p in auto_patterns):
        return 'Automatic'
    if any(re.search(p, text) for p in manual_patterns):
        return 'Manual'

    return None

mask = df['transmission'].isna()
df.loc[mask, 'transmission'] = df.loc[mask, 'description_en'].apply(infer_transmission_from_text)

mask2 = df['transmission'].isna()
df.loc[mask2, 'transmission'] = df.loc[mask2, 'model_version_en'].apply(infer_transmission_from_text)

# ---------------------------------------------------------
# 5. KATMAN – Make / Model mode filling (final)
# ---------------------------------------------------------
ref_cols = ['make', 'model']

mode_map = (
    df.dropna(subset=['transmission'])
      .groupby(ref_cols)['transmission']
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
)

df['transmission'] = df['transmission'].fillna(
    df[ref_cols].apply(lambda r: mode_map.get(tuple(r)), axis=1)
)

# ---------------------------------------------------------
# FINAL KONTROL
# ---------------------------------------------------------
counts, nan_count = value_counts_with_nan(df['transmission'])
print(counts)
print("\nNaN sayısı:", nan_count)

# Eğer hala NaN varsa manuel review için kaydedebilirsin
# manual_df = df[df['transmission'].isna()]
# manual_df.to_excel('Bitirme/transmission_manual_review.xlsx', index=False)


# =============================================================================
# ADIM 16: gears
# =============================================================================

# 1. Geçersiz değerleri NaN yap
invalid_gears = [0, 15, 31, 67, 71, 72, 99]
df.loc[df['gears'].isin(invalid_gears), 'gears'] = np.nan

# 2. Elektrikli araçlar → 1 vites
mask_ev = (df['fuel_category'].str.contains('electric', case=False, na=False)) | df['electric_range_km'].notna()
df.loc[mask_ev & df['gears'].isna(), 'gears'] = 1

# 3. Mode filling: make + model + transmission
gear_ref = (
    df.dropna(subset=['gears'])
      .groupby(['make', 'model', 'transmission'])['gears']
      .agg(lambda x: x.mode().iloc[0])
      .reset_index()
      .rename(columns={'gears': 'gears_ref'})
)
df = df.merge(gear_ref, on=['make', 'model', 'transmission'], how='left')
df['gears'] = df['gears'].fillna(df['gears_ref'])
df.drop(columns='gears_ref', inplace=True)

# 4. Model_version_en ve description_en üzerinden GTRONIC/GT pattern extraction
def extract_gtronic_gears(text):
    if pd.isna(text):
        return np.nan
    text = str(text).upper()
    match = re.search(r'\b([7-9])\s*G\s*-?\s*TRONIC(?=[+\-/\s]|$)|\b([7-9])\s*G\s*T(?=[+\-/\s]|$)', text)
    if match:
        return int(match.group(1) if match.group(1) is not None else match.group(2))
    return np.nan

mask = df['gears'].isna()
df.loc[mask, 'gears'] = df.loc[mask, 'model_version_en'].apply(extract_gtronic_gears)
mask = df['gears'].isna()
df.loc[mask, 'gears'] = df.loc[mask, 'description_en'].apply(extract_gtronic_gears)

# 5. Generic "gear"/"gang" pattern extraction
def extract_gears(text):
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    match = re.search(r'(\d)[-\s]?(gang|gear)', text)
    if match and match.group(1).isdigit():
        return int(match.group(1))
    return np.nan

df['gears_desc'] = df['description_en'].apply(extract_gears)
df['gears'] = df['gears'].fillna(df['gears_desc'])
df.drop(columns='gears_desc', inplace=True)

# 6. Son NaN kontrolü
counts, nan_count = value_counts_with_nan(df['gears'])
print(counts)
print("\nNaN sayısı:", nan_count)


##### MANUEL INSPECTION #####
# gears_df = df[df['gears'].isna()]
# gears_df.to_excel('Bitirme/gears_manual_review.xlsx', index=False)


# =============================================================================
# ADIM 17: drive_train
# =============================================================================

# İlk dağılım
counts, nan_count = value_counts_with_nan(df['drive_train'])
print(counts)
print("\nNaN sayısı:", nan_count)

# ---------------------------------------------------------
# 1. KATMAN: make + model + model_version_en bazlı filling
# ---------------------------------------------------------
mode_map = (
    df.dropna(subset=['drive_train'])
      .groupby(['make', 'model', 'model_version_en'])['drive_train']
      .agg(lambda x: x.mode().iloc[0])
)
mask = df['drive_train'].isna()
df.loc[mask, 'drive_train'] = df.loc[mask].set_index(['make','model','model_version_en']).index.map(mode_map)

# ---------------------------------------------------------
# 2. KATMAN: make + model bazlı filling
# ---------------------------------------------------------
mode_map_2 = (
    df.dropna(subset=['drive_train'])
      .groupby(['make', 'model'])['drive_train']
      .agg(lambda x: x.mode().iloc[0])
)
mask = df['drive_train'].isna()
df.loc[mask, 'drive_train'] = df.loc[mask].set_index(['make','model']).index.map(mode_map_2)

# ---------------------------------------------------------
# 3. KATMAN: description_en ve model_version_en üzerinden regex pattern ile filling
# ---------------------------------------------------------
mask = df['drive_train'].isna()

patterns = {
    '4WD': r'quattro|awd|4x4|4motion|4matic|4Matic',
    'Front Wheel Drive': r'fwd|front wheel',
    'Rear Wheel Drive': r'rwd|rear wheel'
}

for target, pattern in patterns.items():
    df.loc[mask & df['description_en'].str.contains(pattern, case=False, na=False), 'drive_train'] = target
    df.loc[mask & df['model_version_en'].str.contains(pattern, case=False, na=False), 'drive_train'] = target

# ---------------------------------------------------------
# 4. KATMAN: fallback
# ---------------------------------------------------------
df['drive_train'] = df['drive_train'].fillna('Unknown')

# Final kontrol
counts, nan_count = value_counts_with_nan(df['drive_train'])
print(counts)
print("\nNaN sayısı:", nan_count)

# =============================================================================
# ADIM 18: cylinders
# =============================================================================

# Mevcut dağılım ve NaN sayısı
counts, nan_count = value_counts_with_nan(df['cylinders'])
print(counts)
print("\nNaN sayısı:", nan_count)

# Geçersiz / şüpheli değerleri temizle
valid_cylinders = [2, 3, 4, 5, 6, 8, 10, 12, 16]
df.loc[~df['cylinders'].isin(valid_cylinders), 'cylinders'] = np.nan

# 1. KATMAN: make + model bazlı en sık cylinder sayısı ile doldurma
df['cylinders'] = df.groupby(['make', 'model'])['cylinders'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 4)
)

# Final kontrol
counts, nan_count = value_counts_with_nan(df['cylinders'])
print(counts)
print("\nNaN sayısı:", nan_count)


# =============================================================================
# ADIM 19: cylinders_volume_cc
# =============================================================================

# Mevcut dağılım ve NaN sayısı
counts, nan_count = value_counts_with_nan(df['cylinders_volume_cc'])
print(counts)
print("\nNaN sayısı:", nan_count)

# İlk doldurma ve tip düzeltme
df['cylinders_volume_cc'] = df['cylinders_volume_cc'].fillna(0).astype(int)

# Mantıksız değerleri NaN yap
df.loc[(df['cylinders_volume_cc'] < 600) | (df['cylinders_volume_cc'] > 8000), 'cylinders_volume_cc'] = np.nan

# CYLINDERS eksiklerini cc ile doldurma
def cc_to_cylinders(cc):
    if pd.isna(cc):
        return np.nan
    if cc <= 1300:
        return 3
    elif cc <= 1600:
        return 4
    elif cc <= 2000:
        return 4
    elif cc <= 3000:
        return 6
    elif cc <= 5000:
        return 8
    elif cc <= 6000:
        return 12
    else:
        return 16

mask_missing_cyl = df['cylinders'].isna()
df.loc[mask_missing_cyl, 'cylinders'] = df.loc[mask_missing_cyl, 'cylinders_volume_cc'].apply(cc_to_cylinders)

# Final kontrol
counts, nan_count = value_counts_with_nan(df['cylinders'])
print(counts)
print("\nNaN sayısı:", nan_count)



# =============================================================================
# ADIM 20: weight_kg
# =============================================================================
counts, nan_count = value_counts_with_nan(df['weight_kg']); print(counts); print("\nNaN sayısı:", nan_count)

df["weight_kg"] = df["weight_kg"].fillna(0)
len(df[df["weight_kg"]==0])
df["weight_kg"] = (
    df["weight_kg"]
      .astype(str)
      .str.replace(r"[^\d.]", "", regex=True)
      .replace("", np.nan)
      .astype(float)
      .round()
      .astype("float")
)

df['weight_kg'].unique()

#df['weight_kg'] = df['weight_kg'].str.replace(' kg', '').str.replace(',', '').astype(float)

missing_count = df['weight_kg'].isna().sum()
missing_percent = missing_count / len(df) * 100
print(missing_count, missing_percent)


# 1. KATMAN
df.loc[(df['weight_kg'] < 500) | (df['weight_kg'] > 5000), 'weight_kg'] = np.nan

df['weight_kg'].describe()
df['weight_kg'].hist(bins=50)
df['weight_kg'].unique()

# 2. KATMAN
def extract_weight(desc):
    match = re.search(r'(\d{3,5})\s?kg', str(desc))
    return float(match.group(1)) if match else np.nan

df['weight_from_desc'] = df['description_en'].apply(extract_weight)
df['weight_kg'] = df['weight_kg'].fillna(df['weight_from_desc'])


# 3. KATMAN
df['weight_kg_filled'] = df.groupby(['make','model'])['weight_kg'].transform(lambda x: x.fillna(x.median()))
missing_count = df['weight_kg_filled'].isna().sum()
missing_percent = missing_count / len(df) * 100
print(missing_count, missing_percent)

# 4. KATMAN
bins = [0, 1000, 1500, 2000, 2500, 5000]
labels = [1,2,3,4,5]
df['weight_bin'] = pd.cut(df['weight_kg_filled'], bins=bins, labels=labels)


# =============================================================================
# ADIM 21: has_particle_filter
# =============================================================================
# Binary
counts, nan_count = value_counts_with_nan(df['has_particle_filter']); print(counts); print("\nNaN sayısı:", nan_count)
df["has_particle_filter"] = df["has_particle_filter"].astype(int)

# =============================================================================
# ADIM 22: fuel_category
# =============================================================================
counts, nan_count = value_counts_with_nan(df['fuel_category']); print(counts); print("\nNaN sayısı:", nan_count)

# 1. KATMAN
# Grup bazlı mod
group_mode = df.groupby(['make','model','model_version_en'])['fuel_category'] \
               .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Merge ederek NaN olanları doldur
df = df.merge(group_mode.rename('group_mode'), on=['make','model','model_version_en'], how='left')
df['fuel_category_filled'] = df['fuel_category'].fillna(df['group_mode'])

counts, nan_count = value_counts_with_nan(df['fuel_category_filled']); print(counts); print("\nNaN sayısı:", nan_count)

# fuel_category_filled_df = df[df['fuel_category_filled'].isna()]
# fuel_category_filled_df.to_excel('Bitirme/fuel_category_filled_manual_review.xlsx', index=False)

# 2. KATMAN

# Önce tüm model ve model_version_en sütunlarını string yapıyoruz
df['model_version_en'] = df['model_version_en'].astype(str)
df['model'] = df['model'].astype(str)

def infer_fuel(row):
    mv = row['model_version_en'].lower()
    m = row['model'].lower()

    # model_version_en kontrolü
    if 'phev' in mv:
        return 'Electric/Gasoline'
    elif 'tfsie' in mv or 'tfsi e' in mv:
        return 'Electric/Gasoline'
    elif 'hybrid' in mv:
        return 'Electric/Gasoline'
    elif 't5' in mv:
        return 'Gasoline'
    elif 'e-tron' in mv or 'etron' in mv:
        return 'Electric'
    elif any(mv.endswith(suffix) for suffix in ['320i']):  # ix exclude , any number
        return 'Gasoline'

    # model sütunu kontrolü
    if 'e-tron' in m or 'etron' in m:
        return 'Electric'

    return np.nan  # Eğer eşleşmezse NaN bırak

# Eksik fuel_category olanları doldur
#df['fuel_category_filled'] = df['fuel_category']
mask = df['fuel_category_filled'].isna()
df.loc[mask, 'fuel_category_filled'] = df[mask].apply(infer_fuel, axis=1)

counts, nan_count = value_counts_with_nan(df['fuel_category_filled']); print(counts); print("\nNaN sayısı:", nan_count)

# 3. KATMAN

# Hala kalanları global moda doldur
most_common_fuel = df['fuel_category'].mode()[0]
df['fuel_category_filled'] = df['fuel_category_filled'].fillna(most_common_fuel)




# =============================================================================
# primary_fuel
# =============================================================================
# # DROP


# =============================================================================
# ADIM 23: electric_range_km / electric_range_city_km / fuel_cons_comb_l100_km
# =============================================================================

# 0. Make + Model kolonunu oluştur
df['make_model'] = df['make'].astype(str) + " " + df['model'].astype(str)

# Yakıt kategorileri
ice_types = ['Gasoline', 'Diesel']
ev_types = ['Electric']
hybrid_types = ['Electric/Gasoline', 'Electric/Diesel']

# ---------------------------------------------------------
# 1. Menzil (range) doldurma
# ---------------------------------------------------------
cols_range = ['electric_range_km', 'electric_range_city_km']

# 1A. ICE araçlar → range = 0
mask_ice = df['fuel_category'].isin(ice_types)
for col in cols_range:
    if col in df.columns:
        df.loc[mask_ice & df[col].isna(), col] = 0

# 1B. Hibrit araçlar → make_model medyan + kalan 0
mask_hybrid = df['fuel_category'].astype(str).str.contains('Electric/', na=False)
for col in cols_range:
    if col in df.columns:
        # Medyan ile doldurma
        df.loc[mask_hybrid, col] = df.loc[mask_hybrid, col].fillna(
            df.loc[mask_hybrid].groupby('make_model')[col].transform('median')
        )
        # Kalan boşluklar → 0
        df.loc[mask_hybrid & df[col].isna(), col] = 0

# 1C. Saf elektrikli araçlar → make_model medyan + global medyan
mask_ev = df['fuel_category'].isin(ev_types)
for col in cols_range:
    if col in df.columns:
        df.loc[mask_ev, col] = df.loc[mask_ev, col].fillna(
            df.loc[mask_ev].groupby('make_model')[col].transform('median')
        )
        df.loc[mask_ev, col] = df.loc[mask_ev, col].fillna(df.loc[mask_ev, col].median())

# ---------------------------------------------------------
# 2. Fuel consumption doldurma
# ---------------------------------------------------------
if 'fuel_cons_comb_l100_km' in df.columns:
    # Saf elektrikli → 0
    df.loc[mask_ev & df['fuel_cons_comb_l100_km'].isna(), 'fuel_cons_comb_l100_km'] = 0

    # Diğer araçlar → make_model medyan + global medyan
    fuel_users = ice_types + hybrid_types
    mask_fuel = df['fuel_category'].isin(fuel_users)
    df.loc[mask_fuel, 'fuel_cons_comb_l100_km'] = df.loc[mask_fuel, 'fuel_cons_comb_l100_km'].fillna(
        df.loc[mask_fuel].groupby('make_model')['fuel_cons_comb_l100_km'].transform('median')
    )
    df['fuel_cons_comb_l100_km'].fillna(df.loc[mask_fuel, 'fuel_cons_comb_l100_km'].median(), inplace=True)

# ---------------------------------------------------------
# 3. Equipment kolonları boşları temizleme
# ---------------------------------------------------------
equip_cols = ['equipment_comfort', 'equipment_entertainment', 'equipment_extra', 'equipment_safety']
for col in equip_cols:
    if col in df.columns:
        df[col].fillna("", inplace=True)

# ---------------------------------------------------------
# 4. Final kontrol: kalan boşları 0 yap
# ---------------------------------------------------------
for col in cols_range:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

print("Electric range ve fuel consumption doldurma tamamlandı.")


# =============================================================================
# Equipment (equipment_comfort / equipment_entertainment / equipment_extra / equipment_safety)
# =============================================================================
 # Bu bölüm, feature engineering aşamasında ele alınacak.
# Şu an için EDA ve modelleme açısından herhangi bir sorun gözlemlemedik.

# Stratejimiz: “Her özellik ayrı sütunda”

# Temizlik: Önce, özelliklerin başındaki ve sonundaki boşlukları kaldıracağız.
# Böylece örneğin "ABS" ile " ABS" farklı algılanmayacak.

# Ayırma (Splitting): Özellikleri virgüllerle ayırıp her birini ayrı sütuna dönüştüreceğiz.

# Önek (Prefix) ekleme: Hangi özelliğin hangi gruptan geldiğini belli etmek için
# sütun isimlerinin başına grup adını ekleyeceğiz. Örneğin comfort_AirConditioning.

# Birleştirme: Oluşan 0-1 sütunlarını ana veri setine ekleyip, orijinal metin sütununu kaldıracağız.

# Bu işlem, çok etiketli veriler (Multi-Label) için uygundur.
# Örneğin "Klima, ABS" ve "Klima, ABS, Sunroof" gibi kombinasyonlar tek bir sınıf olarak ele alınmaz.
# Her özellik kendi sütununda 1 veya 0 ile temsil edilir, böylece binlerce gereksiz sütun oluşmaz.



# =============================================================================
# True/False Columns
# =============================================================================
bool_cols = [
    "is_used",
    "is_new",
    "is_preregistered",
    "has_full_service_history",
    "non_smoking",
    "is_rental",
    "seller_is_dealer"
]

# Boolean değerleri 0/1 formatına çevir
df[bool_cols] = df[bool_cols].astype(int)

# =============================================================================
# ADIM 24: is_used / is_new
# =============================================================================

# 1. nr_prev_owners = 0 ise is_new güncelleme
df.loc[
    (df["is_new"] == 0) &
    (df["is_used"] == 0) &
    (df["nr_prev_owners"] == 0),
    "is_new"
] = 1

# 2. nr_prev_owners <= 1 ve mileage < 120 km ise is_new güncelleme
df.loc[
    (df["is_new"] == 0) &
    (df["is_used"] == 0) &
    (df["nr_prev_owners"] <= 1) &
    (df["mileage_km_raw"] < 120),
    "is_new"
] = 1

# 3. Kalan boş satırlar için is_used = 1
df.loc[
    (df["is_new"] == 0) &
    (df["is_used"] == 0),
    "is_used"
] = 1


# Sonuç: Yukarıdaki dört koşulun hepsini sağlayan satırların 'is_new' değeri 1 olarak güncellenmiştir.
# no_prev_owners <= 1 iken mileage_km_raw <120

# =============================================================================
# ADIM 25: is_preregistered
# =============================================================================

# True olanları çıkar (çok dengesiz)
df = df[df["is_preregistered"] != True]
df = df.drop(columns=['is_preregistered'])


# =============================================================================
# ADIM 26: had_accident
# =============================================================================

# True olanları çıkar
df = df[df["had_accident"] != True]
df = df.drop(columns=['had_accident'])


# =============================================================================
# ADIM 27: has_full_service_history
# =============================================================================

df["has_full_service_history"].value_counts(normalize=True) * 100


# =============================================================================
# ADIM 28: non_smoking
# =============================================================================

df["non_smoking"].value_counts(normalize=True) * 100


# =============================================================================
# ADIM 29: nr_prev_owners
# =============================================================================

# 5'ten büyük sayısal değerleri 'Multiple_Owners' olarak işaretle
numeric_mask = pd.to_numeric(df['nr_prev_owners'], errors='coerce').notna()
df.loc[
    numeric_mask & (pd.to_numeric(df['nr_prev_owners'], errors='coerce') > 5.0),
    'nr_prev_owners'
] = 'Multiple_Owners'

# Kategori tipine çevir
df['nr_prev_owners'] = df['nr_prev_owners'].astype('category')

# Eksikleri 0 ile doldur
df['nr_prev_owners'] = df['nr_prev_owners'].fillna(0.0)

# Değer dağılımı
df['nr_prev_owners'].value_counts()



# =============================================================================
# ADIM 30: is_rental
# =============================================================================

# Oran kontrolü
df["is_rental"].value_counts(normalize=True) * 100

# True olanları çıkar
df = df[df["is_rental"] != True]

# Sütunu düşür
df = df.drop(columns=['is_rental'])



# =============================================================================
# ADIM 31: envir_standard
# =============================================================================

# Eksikleri üretim yılına göre doldur
mask = df["envir_standard"].isna()
df.loc[mask, "envir_standard"] = df.loc[mask, "production_year"].apply(
    lambda y: "Euro 6" if pd.notna(y) and y >= 2015 else "Euro 5"
)

# Dağılım kontrolü
df["envir_standard"].value_counts(dropna=False)

# Euro 6 grubu oluştur
df["euro6_group"] = np.where(
    df["envir_standard"].str.startswith("Euro 6", na=False),
    "Euro 6",
    "Other"
)

df["euro6_group"].value_counts(normalize=True) * 100


# =============================================================================
# ADIM 32: offer_type
# =============================================================================

# Tekrar dağılım kontrolü
df["offer_type"].value_counts(normalize=True) * 100

# Binary dönüşüm: U → 1, N → 0
df["offer_type_bin"] = df["offer_type"].map({"U": 1, "N": 0})
df["offer_type_bin"].value_counts(normalize=True) * 100


# =============================================================================
# ADIM 33: country_code
# =============================================================================

df["country_code"].value_counts()

# Popüler ülkeler vs diğerleri
top_countries = ["DE", "IT", "NL"]
df["is_top_country"] = df["country_code"].isin(top_countries).astype(int)


# =============================================================================
# ADIM 34: seller_is_dealer
# =============================================================================

df["seller_is_dealer"].value_counts(normalize=True) * 100


# ==================================================================================================================
# Final check for EDA and Feature Improvement (Raw Features)
# ==================================================================================================================

# Kullanılmayacak sütunları düşür
drop_cols_1 = [
    'id', 'description_en', 'ratings_count', 'ratings_recommend_percentage',
    'price_currency', 'price_net', 'price_vat_rate', 'vin', 'german_hsn_tsn',
    'mileage_km', 'registration_date', 'vehicle_type', 'body_color_original_en',
    'upholstery_color', 'cylinders', 'weight_kg', 'primary_fuel',
    'fuel_cons_city_l100_km', 'fuel_cons_highway_l100_km',
    'co2_emission_grper_km', 'fuel_cons_comb_l100_wltp_km',
    'fuel_cons_electric_comb_l100_wltp_km', 'co2_emission_grper_wltp_km',
    'is_new', 'original_market',
    'zip', 'city', 'street', 'latitude', 'longitude',
    'seller_type', 'seller_company_name', 'has_warranty', 'warranty'
]
df = df.drop(columns=drop_cols_1)

# Eksik değer kontrolü
df.isnull().sum()

# Ara sütunları düşür
drop_cols_2 = ['model_fill_source', 'power_kw_bin', 'group_mode']
df = df.drop(columns=drop_cols_2)

# Kalan eksik değerler nedeniyle bir sütunu daha düşür
df = df.drop(columns=['cylinders_volume_cc'])

# Eksik değer kontrolü
df.isnull().sum()
df.columns

# Kayıtları eksiksiz hale getir
df = df.dropna()
df.to_csv("auto_processed_after_drop.csv")

# Son durum
df.shape


# =============================================================================
# Son durumda Target Değişkene Göre Analiz
# =============================================================================

# Kategorik değişkenler için target özeti
def target_summary_with_cat(dataframe, target, categorical_col):
    print(f"### {categorical_col} ###")
    summary_df = pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
        "Count": dataframe[categorical_col].value_counts(),
        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)
    })
    print(summary_df, end="\n\n\n")


# Sayısal değişkenler için target özeti
def target_summary_with_num(dataframe, target, numerical_col):
    print(f"### {numerical_col} ###")
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Örnek: cat_cols ve num_cols tanımlı olsun
cat_cols = ["transmission", "fuel_category", "drive_train", "make"]  # örnek
num_cols = ["power_kw", "mileage_km_raw", "weight_kg_filled"]        # örnek

# Target değişken
target_col = "price"  # veya df'deki gerçek target değişkenin

# Kategorik değişkenleri incele
for col in cat_cols:
    target_summary_with_cat(df, target_col, col)

# Sayısal değişkenleri incele
for col in num_cols:
    target_summary_with_num(df, target_col, col)