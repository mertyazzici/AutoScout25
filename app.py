import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# -----------------------------------------------------------------------------
# 1. SAYFA AYARLARI VE TASARIM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoScout25 Fiyat Tahmini",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile biraz makyaj yapalım (Tablo kenarları, buton renkleri vb.)
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-size: 20px;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODELİ YÜKLE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_data_and_model():
    try:
        metadata = joblib.load("model_metadata.pkl")
        model = CatBoostRegressor()
        model.load_model("catboost_car_price_model.cbm")
        return model, metadata
    except Exception as e:
        st.error(f"Dosyalar yüklenirken hata oluştu: {e}")
        return None, None


model, metadata = load_data_and_model()

if model is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR (KULLANICI GİRİŞLERİ)
# -----------------------------------------------------------------------------
st.sidebar.title("Araç Konfigüratörü")
st.sidebar.markdown("Aracın özelliklerini aşağıdan seçiniz.")

user_input = {}

# --- BÖLÜM 1: TEMEL BİLGİLER (Marka, Model, Yıl) ---
st.sidebar.subheader("Temel Bilgiler")

# 1.1 Marka Seçimi
makes = sorted(metadata["make_model_map"].keys())
selected_make = st.sidebar.selectbox("Marka", makes, index=makes.index("Opel") if "Opel" in makes else 0)
user_input["make"] = selected_make

# 1.2 Model Seçimi (Markaya Göre Filtreli)
available_models = sorted(metadata["make_model_map"][selected_make])
selected_model = st.sidebar.selectbox("Model", available_models)
user_input["model"] = selected_model

# 1.3 Diğer Temel Bilgiler
user_input["body_type"] = st.sidebar.selectbox("Kasa Tipi",
                                               sorted([str(x) for x in metadata["cat_options"]["body_type"]]))
user_input["production_year"] = st.sidebar.number_input("Üretim Yılı", 1990, 2025, 2020)
user_input["mileage_km_raw"] = st.sidebar.number_input("Kilometre", 0, 1000000, 50000, step=5000)

# --- BÖLÜM 2: TEKNİK DETAYLAR (Motor, Vites) ---
with st.sidebar.expander("⚙️ Motor ve Performans", expanded=False):
    user_input["transmission"] = st.selectbox("Vites Tipi",
                                              sorted([str(x) for x in metadata["cat_options"]["transmission"]]))
    user_input["fuel_category"] = st.selectbox("Yakıt Tipi",
                                               sorted([str(x) for x in metadata["cat_options"]["fuel_category"]]))
    user_input["power_kw"] = st.number_input("Motor Gücü (kW)", 0, 800, 100)
    user_input["gears"] = st.slider("Vites Sayısı", 1, 10, 6)
    user_input["fuel_cons_comb_l100_km"] = st.number_input("Ort. Yakıt (l/100km)", 0.0, 30.0, 6.5)

# --- BÖLÜM 3: DONANIM VE DURUM ---
with st.sidebar.expander("Donanım ve Durum", expanded=False):
    col1, col2 = st.columns(2)
    user_input["body_color"] = st.selectbox("Renk", sorted([str(x) for x in metadata["cat_options"]["body_color"]]))
    user_input["upholstery"] = st.selectbox("Döşeme", sorted([str(x) for x in metadata["cat_options"]["upholstery"]]))

    user_input["nr_seats"] = st.slider("Koltuk", 2, 9, 5)
    user_input["nr_doors"] = st.slider("Kapı", 2, 5, 5)

    # Checkbox benzeri Boolean/Binary değerler
    user_input["is_used"] = "Yes"  # Varsayılan
    user_input["seller_is_dealer"] = "Yes"  # Varsayılan
    # Diğer gerekli kategoriklerin varsayılanlarını en çok geçen (mode) veya ilk değer ile dolduralım
    # Kullanıcıyı yormamak için bazılarını arka planda sabitliyoruz veya listelerden seçtiriyoruz
    for col in metadata["cat_cols"]:
        if col not in user_input:
            # Eğer yukarıda elle eklemediysek, listedeki ilk değeri al
            user_input[col] = metadata["cat_options"][col][0]

    # Eksik numerik alanları doldur (Elektrikli araç değilse 0 gibi)
    user_input["electric_range_km"] = 0
    user_input["electric_range_city_km"] = 0
    user_input["nr_prev_owners"] = 1

# -----------------------------------------------------------------------------
# 4. ANA EKRAN (GÖRSELLEŞTİRME VE SONUÇ)
# -----------------------------------------------------------------------------

# Başlık
st.title("🚗 Fiyat Tahmin Asistanı")
st.markdown("---")

# Seçilen aracın kısa özeti (Kart Görünümü)
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Marka", user_input["make"])
col_info2.metric("Model", user_input["model"])
col_info3.metric("Yıl", user_input["production_year"])
col_info4.metric("KM", f"{user_input['mileage_km_raw']:,}")

st.markdown("---")

# TAHMİN BUTONU VE SONUÇ
if st.button("Fiyatı Hesapla"):

    # 1. Veriyi DataFrame'e çevir
    df_input = pd.DataFrame([user_input])

    # 2. Sütun sırasını eşle
    df_input = df_input.reindex(columns=metadata["columns_order"])

    # 3. Sayısal dönüşümleri yap
    for col in metadata["num_cols"]:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

    # 4. Yüklenme efekti
    with st.spinner('Yapay zeka aracı analiz ediyor...'):
        prediction_log = model.predict(df_input)[0]
        prediction_price = np.expm1(prediction_log)

    # 5. Sonucu Büyük Göster
    st.markdown(f"""
    <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center;">
        <h3 style="color: #155724; margin:0;">Tahmini Piyasa Değeri</h3>
        <h1 style="color: #155724; font-size: 60px; margin:0;">{prediction_price:,.0f} €</h1>
        <p style="color: #155724;">Bu fiyat piyasa koşullarına göre değişiklik gösterebilir.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Sol menüden araç özelliklerini seçip 'Fiyatı Hesapla' butonuna basınız.")