import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import random

# =============================================================================
# 1. AYARLAR: KİM HANGİ KATEGORİDE?
# =============================================================================

# Dosya Adı
CSV_FILE = "auto_processed_after_drop.csv"

# HYPER CAR: Asla modele sokma, direkt manuel fiyatla/ortalamayla yönet.
HYPER_MAKES = ["Bugatti", "Pagani", "Koenigsegg"]

# SUPER CAR: Ayrı bir model hak eden lüks devler.
SUPER_MAKES = [
    "Ferrari", "Lamborghini", "McLaren", "Rolls-Royce",
    "Bentley", "Aston Martin", "Maserati", "Maybach"
]

# =============================================================================
# 2. VERİ OKUMA VE TEMİZLİK
# =============================================================================

if not os.path.exists(CSV_FILE):
    # Belki bir alt klasördedir diye kontrol edelim
    if os.path.exists(f"FinaProje2025/{CSV_FILE}"):
        CSV_FILE = f"FinaProje2025/{CSV_FILE}"
    else:
        print(f" '{CSV_FILE}' bulunamadı. Dosya adını kontrol et.")
        exit()

df = pd.read_csv(CSV_FILE, low_memory=False)
upholstery_map = {
    "alcantara": "Alcantara",
    "Others": "Other",
    "Leather": "Full leather"
}

if "upholstery" in df.columns:
    df["upholstery"] = df["upholstery"].replace(upholstery_map)

print(f" {len(df)} satır yüklendi.")

# Fiyat temizliği
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])  # Fiyatı olmayanları at

# Kategorik temizlik (Büyük/Küçük harf, boşluk)
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# Log Dönüşümü (Modeller için)
df["price_log"] = np.log1p(df["price"])


# =============================================================================
# 3. SEGMENTASYON (3 KATMANLI YAPI)
# =============================================================================
def define_segment(make):
    if make in HYPER_MAKES:
        return "Hyper"
    elif make in SUPER_MAKES:
        return "Super"
    else:
        return "Standard"


df["segment"] = df["make"].apply(define_segment)

# Veriyi Parçala
df_hyper = df[df["segment"] == "Hyper"].copy()
df_super = df[df["segment"] == "Super"].copy()
df_standard = df[df["segment"] == "Standard"].copy()

print(
    f" Veri Dağılımı:\n    Hyper (Bugatti): {len(df_hyper)}\n    Super (Ferrari vb.): {len(df_super)}\n   Standard (Halk): {len(df_standard)}")

# =============================================================================
# 4. HYPER CAR İŞLEMLERİ (MANUEL KURAL SETİ)
# =============================================================================
# Hyper car için model eğitmiyoruz. İstatistiklerini kaydediyoruz.
hyper_stats = {}
if not df_hyper.empty:
    for make in df_hyper["make"].unique():
        avg_price = df_hyper[df_hyper["make"] == make]["price"].mean()
        # Güvenlik önlemi: Eğer ortalama çok düşük çıkarsa (veri hatası varsa) en az 1.5M olsun
        base_price = max(1_500_000, avg_price)
        hyper_stats[make] = {
            "avg_price": avg_price,
            "base_price": base_price
        }
    print(f"Hyper Car İstatistikleri: {list(hyper_stats.keys())}")
else:
    print("Veri setinde hiç Hyper Car bulunamadı.")

# =============================================================================
# 5. MODEL EĞİTİM FONKSİYONU
# =============================================================================
# Modelde kullanılmayacak sütunlar
DROP_COLS = [
    "price", "price_log", "segment", "ratings_average", "model_version_en",
    "country_code", "offer_type_bin", "NEW_IS_RATED", "make_model", "model_key"
]

# Sadece df'de var olanları filtrele
cols_to_drop = [c for c in DROP_COLS if c in df.columns]


def train_segment_model(dataframe, segment_name):
    if len(dataframe) < 10:
        print(f" {segment_name} için yeterli veri yok, eğitim atlandı.")
        return None, None, []

    X = dataframe.drop(columns=cols_to_drop, errors='ignore')
    y = dataframe["price_log"]

    # ---------------------------------------------------------
    # OTOMATİK TİP TANIMA (Hata almamak için)
    # ---------------------------------------------------------
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # NaN Doldurma
    for col in X.columns:
        if col in cat_features:
            X[col] = X[col].astype(str).replace('Nan', 'Unknown').fillna("Unknown")
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Train/Test Spliti
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



    model = CatBoostRegressor(
        iterations=2000,  # İstediğin iterasyon sayısı
        learning_rate=0.03,  # İstediğin öğrenme oranı
        depth=6,  # İstediğin derinlik
        random_seed=42,
        allow_writing_files=False,
        verbose=100,  # İlerlemeyi göster
        cat_features=cat_features,
        loss_function='RMSE'
    )

    print(f"\n======== {segment_name} Modeli Eğitiliyor ({len(X)} satır veri ile) ========")
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    # ---------------------------------------------------------
    # METRİK HESAPLAMA (GERÇEK FİYATLAR ÜZERİNDEN)
    # ---------------------------------------------------------
    print(f"\n--- {segment_name} Validasyon Performansı ---")

    # Validation seti üzerinde tahmin yap
    y_pred_log = model.predict(X_val)

    # Log dönüşümünü geri al (np.expm1 çünkü np.log1p kullanmıştık)
    y_true_orig = np.expm1(y_val)
    y_pred_orig = np.expm1(y_pred_log)

    # Metrikleri hesapla
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2_log = r2_score(y_val, y_pred_log)

    print(f"RMSE (Ortalama Karekök Hata): {rmse:,.2f}")
    print(f"MAE  (Ortalama Mutlak Hata) : {mae:,.2f}")
    print(f"R2 Score (Açıklayıcılık)    : {r2_log:.4f}")
    print("-" * 50)

    # Kaydet
    filename = f"catboost_{segment_name.lower()}.cbm"
    model.save_model(filename)
    print(f" {filename} başarıyla kaydedildi.")

    return model, X.columns.tolist(), cat_features


# =============================================================================
# 6. EĞİTİMİ BAŞLAT
# =============================================================================

# 1. Super Car Eğitimi
model_super, cols_super, cat_super = train_segment_model(df_super, "Super")

# 2. Standard Car Eğitimi
model_std, cols_std, cat_std = train_segment_model(df_standard, "Standard")

# =============================================================================
# 7. METADATA KAYDETME
# =============================================================================
# Make-Model haritası (Tüm veri üzerinden - Bugatti dahil)
make_model_map = df.groupby("make")["model"].unique().apply(list).to_dict()

# Kategori listelerini (Dropdown seçenekleri) Standard modelden alalım
cat_options = {}
if cols_std:
    # Veri tiplerini koruyarak seçenekleri al
    X_temp = df_standard.drop(columns=cols_to_drop, errors='ignore')
    for col in cat_std:
        if col in X_temp.columns:
            cat_options[col] = X_temp[col].astype(str).unique().tolist()

metadata = {
    "columns_order_std": cols_std,  # Standard model kolon sırası
    "columns_order_super": cols_super,  # Super model kolon sırası
    "cat_cols": cat_std,  # Kategorik kolonlar
    "num_cols": [c for c in cols_std if c not in cat_std] if cols_std else [],
    "make_model_map": make_model_map,  # Marka-Model ilişkisi
    "cat_options": cat_options,  # Dropdown seçenekleri
    "hyper_makes": HYPER_MAKES,  # Hyper markaları listesi
    "super_makes": SUPER_MAKES,  # Super markaları listesi
    "hyper_stats": hyper_stats  # Bugatti fiyat kuralları
}

joblib.dump(metadata, "model_metadata_3tier.pkl")
print("\nMetadata kaydedildi: model_metadata_3tier.pkl")