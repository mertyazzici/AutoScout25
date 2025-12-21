import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
import os

# -----------------------------------------------------------------------------
# 1. SAYFA AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoScout25 | DropNA Edition",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. PREMIUM CSS TASARIMI (Midnight Navy & Gold + Segment Kartları)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Font: Montserrat */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    /* --- ANA GÖVDE RENKLERİ --- */
    .stApp {
        background-color: #0b1426; /* Deep Navy */
        color: #e2e8f0;
    }

    /* --- SIDEBAR (YAN MENÜ) --- */
    [data-testid="stSidebar"] {
        background-color: #080f1f;
        border-right: 1px solid #cfa86050;
    }

    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] label {
        color: #d1d5db !important;
    }

    /* Sidebar Başlık */
    .sidebar-brand {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        color: #cfa860;
        letter-spacing: 2px;
        margin-bottom: 10px;
        text-shadow: 0px 0px 10px rgba(207, 168, 96, 0.3);
    }

    /* Kullanım Kılavuzu */
    .guide-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #cfa860;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
    }
    .guide-step { font-weight: bold; color: #fff; }

    /* --- HEADER --- */
    .premium-header {
        background: linear-gradient(90deg, #162447 0%, #1f4068 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border-bottom: 3px solid #cfa860;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 2px;
        margin: 0;
    }

    /* Beta Yazısı Stili */
    .beta-badge {
        font-size: 0.5em; 
        font-weight: 400;
        color: #cfa860; /* Altın rengi */
        vertical-align: middle;
        opacity: 0.8;
        margin-left: 10px;
        letter-spacing: 1px;
    }

    .header-subtitle {
        color: #cfa860;
        font-size: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 5px;
    }

    /* --- INPUT ALANLARI RENKLENDİRME --- */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #cfa860 !important;
        font-weight: 600;
    }

    /* --- BUTON TASARIMI --- */
    .stButton>button {
        background: linear-gradient(135deg, #cfa860 0%, #b08d55 100%);
        color: #0b1426;
        border: none;
        height: 3.5em;
        font-size: 18px;
        font-weight: 700;
        border-radius: 8px;
        width: 100%;
        transition: 0.3s;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(207, 168, 96, 0.5);
    }

    /* --- SEGMENTLERE ÖZEL KART TASARIMLARI --- */

    /* 1. HYPER CAR KARTI (Siyah & Altın) */
    .card-hyper { 
        background: linear-gradient(135deg, #141E30, #243B55); 
        color: #FFD700; 
        padding: 40px; 
        border-radius: 15px; 
        text-align: center; 
        border: 2px solid #FFD700; 
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.4);
        margin-top: 20px;
    }
    .card-hyper h1 { color: #FFD700; font-size: 4rem; text-shadow: 2px 2px 4px #000; margin: 10px 0; }
    .card-hyper h3 { color: #fff; opacity: 0.9; letter-spacing: 2px; }

    /* 2. SUPER CAR KARTI (Koyu Kırmızı & Siyah) */
    .card-super { 
        background: linear-gradient(135deg, #8E0E00, #1F1C18); 
        color: white; 
        padding: 40px; 
        border-radius: 15px; 
        text-align: center; 
        border: 1px solid #ff4d4d;
        box-shadow: 0 0 25px rgba(255, 0, 0, 0.4);
        margin-top: 20px;
    }
    .card-super h1 { color: #FFF; font-size: 4rem; margin: 10px 0; }
    .card-super h3 { color: #ffcccc; opacity: 0.9; letter-spacing: 2px; }

    /* 3. STANDARD CAR KARTI (DropNA Stili - Lacivert & Altın) */
    .card-std { 
        background-color: #162447;
        border: 1px solid #cfa860;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .card-std h1 { font-size: 4rem; color: #fff; font-weight: 300; margin: 10px 0; }
    .card-std h3 { color: #8da9c4; letter-spacing: 1px; text-transform: uppercase; }
    .currency-symbol { color: #cfa860; font-weight: 600; }

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. DOSYALARI VE MODELLERİ YÜKLEME (3-Tier Yapısı)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    # Dosyaların bulunduğu klasörü kontrol et (Opsiyonel güvenlik)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    path_meta = os.path.join(current_dir, "model_metadata_3tier.pkl")
    path_std = os.path.join(current_dir, "catboost_standard.cbm")
    path_super = os.path.join(current_dir, "catboost_super.cbm")

    try:
        if not os.path.exists(path_meta):
            st.error(f"Dosya bulunamadı: {path_meta}. Lütfen 'new2' klasöründe çalıştığınızdan emin olun.")
            return None, None, None

        meta = joblib.load(path_meta)

        m_std = CatBoostRegressor()
        m_std.load_model(path_std)

        m_super = None
        if os.path.exists(path_super):
            try:
                temp_super = CatBoostRegressor()
                temp_super.load_model(path_super)
                m_super = temp_super
            except:
                pass

        return m_std, m_super, meta
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None

model_std, model_super, metadata = load_assets()

if not model_std:
    st.stop()  # Model yoksa uygulamayı durdur

# -----------------------------------------------------------------------------
# 4. SIDEBAR VE SEGMENT MANTIĞI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">DropNA</div>', unsafe_allow_html=True)

    if os.path.exists("dropna_team.jpeg"):
        st.image("dropna_team.jpeg", use_container_width=True)
    else:
        st.warning("Görsel bulunamadı! 'dropna_team.jpeg' bekleniyor.")

    st.write("---")

    st.markdown("""
    <div class="guide-box">
        <span class="guide-step">🚀 Nasıl Kullanılır?</span>
        <ol style="padding-left:15px; margin-top:10px; color:#d1d5db;">
            <li>Sağdaki panelden <b>Marka ve Modeli</b> seçin.</li>
            <li>Aracın <b>Teknik Özelliklerini</b> girin.</li>
            <li><b>"Değerleme Yap"</b> butonuna basarak sonucu görün.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    with st.expander("ℹ️ Sistem Hakkında"):
        st.markdown("Bu proje **DropNA** veri bilimi ekibi tarafından geliştirilmiştir.")

# -----------------------------------------------------------------------------
# 5. ANA EKRAN VE GİRİŞLER
# -----------------------------------------------------------------------------

# Header
st.markdown("""
    <div class="premium-header">
        <h1 class="header-title">
            🏎️ AUTOSCOUT25 <span class="beta-badge">(Beta)</span>
        </h1>
        <div class="header-subtitle">EXCLUSIVE VEHICLE VALUATION</div>
    </div>
""", unsafe_allow_html=True)

# -- MARKA SEÇİMİ VE SEGMENT BELİRLEME --
all_makes = sorted(metadata["make_model_map"].keys())
default_idx = all_makes.index("Audi") if "Audi" in all_makes else 0
# Sol kolonda kullanmak üzere değişkene alıyoruz ama burada mantığı kuruyoruz
selected_make_temp = all_makes[default_idx]

user_input = {}
col1, col2, col3 = st.columns([1, 1, 1])

# --- SOL KOLON ---
with col1:
    st.markdown("### ARAÇ KİMLİĞİ")
    selected_make = st.selectbox("Marka", all_makes, index=default_idx)
    user_input["make"] = selected_make

    # Segment Bilgisi Gösterimi (Anlık tepki için)
    if selected_make in metadata.get("hyper_makes", []):
        segment = "Hyper"
        st.info(f"💎 **{selected_make}**: Hyper Car (Özel Koleksiyon)")
    elif selected_make in metadata.get("super_makes", []):
        segment = "Super"
        st.warning(f" **{selected_make}**: Super Car (Lüks Segment)")
    else:
        segment = "Standard"
        # Standard için ekstra uyarıya gerek yok, temiz kalsın

    available_models = sorted(metadata["make_model_map"][selected_make])
    selected_model = st.selectbox("Model", available_models)
    user_input["model"] = selected_model

    user_input["production_year"] = st.number_input("Model Yılı", 1990, 2025, 2020)
    user_input["body_type"] = st.selectbox("Kasa Tipi", sorted([str(x) for x in metadata["cat_options"]["body_type"]]))

# --- ORTA KOLON ---
with col2:
    st.markdown("### TEKNİK VERİLER")
    user_input["mileage_km_raw"] = st.number_input("Kilometre", 0, 1000000, 50000, step=5000)

    # Segment'e göre varsayılan KW ayarı
    default_kw = 100
    if segment == "Super": default_kw = 400
    if segment == "Hyper": default_kw = 1100

    user_input["power_kw"] = st.number_input("Motor Gücü (kW)", 0, 1600, default_kw)

    user_input["transmission"] = st.selectbox("Vites Tipi",
                                              sorted([str(x) for x in metadata["cat_options"]["transmission"]]))
    user_input["fuel_category"] = st.selectbox("Yakıt",
                                               sorted([str(x) for x in metadata["cat_options"]["fuel_category"]]))

# --- SAĞ KOLON ---
with col3:
    st.markdown("### DONANIM")

    # 1. VİTES SAYISI (ÖZEL ETİKET + UYARI)
    st.markdown("""
        <div style="margin-bottom: 5px;">
            <span style="color:#cfa860; font-weight:600; font-size:14px;">Vites Sayısı</span>
            <span style="color:#ef4444; font-size:12px; margin-left:8px; font-weight:500;">
                ⚠️ DİKKAT: Elektrikli araçlarda vites sayısını lütfen 1 olarak seçiniz.
            </span>
        </div>
    """, unsafe_allow_html=True)

    user_input["gears"] = st.slider(
        "Vites Sayısı",
        min_value=1,
        max_value=10,
        value=6,
        label_visibility="collapsed"
    )

    # 2. Renk
    colors = sorted([str(x) for x in metadata["cat_options"]["body_color"]])
    user_input["body_color"] = st.selectbox("Renk", colors, index=colors.index("Black") if "Black" in colors else 0)

    # 3. Döşeme
    upholsteries = sorted([str(x) for x in metadata["cat_options"]["upholstery"]])
    user_input["upholstery"] = st.selectbox("Döşeme", upholsteries, index=0)

    # Gizli Varsayılan Değerler (Eksik kolonlar için)
    user_input["fuel_cons_comb_l100_km"] = 5.0
    user_input["nr_seats"] = 5
    user_input["nr_doors"] = 5
    user_input["is_used"] = "Yes"
    user_input["seller_is_dealer"] = "Yes"
    user_input["electric_range_km"] = 0
    user_input["electric_range_city_km"] = 0
    user_input["nr_prev_owners"] = 1

    for col in metadata["cat_cols"]:
        if col not in user_input:
            user_input[col] = metadata["cat_options"][col][0]

    st.write("")
    st.write("")
    predict_btn = st.button("DEĞERLEME YAP ➤")

# -----------------------------------------------------------------------------
# 6. HESAPLAMA VE SONUÇ (3-TIER MANTIĞI)
# -----------------------------------------------------------------------------
if predict_btn:

    final_price = 0
    calculated = False
    error_message = ""

    # --------------------------------------
    # SENARYO 1: HYPER CAR (KURAL BAZLI)
    # --------------------------------------
    if segment == "Hyper":
        stats = metadata.get("hyper_stats", {}).get(selected_make)
        if stats:
            base_price = stats["base_price"]
        else:
            base_price = 2_500_000  # Fallback

        age = 2026 - user_input["production_year"]
        depreciation = (age * 0.01 * base_price) + (user_input["mileage_km_raw"] * 5)
        final_price = max(base_price * 0.85, base_price - depreciation)
        calculated = True

    # --------------------------------------
    # SENARYO 2 & 3: SUPER & STANDARD (MODEL)
    # --------------------------------------
    else:
        # Model ve Kolon Sırası Seçimi
        if segment == "Super":
            active_model = model_super
            cols_needed = metadata.get("columns_order_super", [])
        else:
            active_model = model_std
            cols_needed = metadata.get("columns_order_std",
                                       [])  # Eski metadata uyumu için 'std' anahtarı kontrol edilmeli

        # Eğer Super model yoksa veya kolonlar eksikse hata ver
        if active_model is None and segment == "Super":
            error_message = "Bu segment için Super Car modeli yüklenemedi."
        elif not cols_needed:
            # Eski metadata kullanılıyorsa "columns_order" anahtarına bak
            cols_needed = metadata.get("columns_order", [])
            if not cols_needed:
                error_message = "Model kolon sıralaması bulunamadı."
            else:
                active_model = model_std  # Fallback olarak standart modeli kullan

        if not error_message:
            # DataFrame Hazırla
            df_input = pd.DataFrame([user_input])
            df_input = df_input.reindex(columns=cols_needed)

            # Tip Dönüşümleri
            for col in df_input.columns:
                if col in metadata["cat_cols"]:
                    df_input[col] = df_input[col].astype(str)
                else:
                    df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0)

            # Tahmin
            with st.spinner('Piyasa analizi yapılıyor...'):
                try:
                    prediction_log = active_model.predict(df_input)[0]
                    final_price = np.expm1(prediction_log)
                    calculated = True
                except Exception as e:
                    error_message = f"Hesaplama hatası: {e}"

    # --------------------------------------
    # SONUÇ GÖRÜNTÜLEME
    # --------------------------------------
    if calculated:
        st.markdown("---")

        if segment == "Hyper":
            # BUGATTI TARZI KART
            st.markdown(f"""
                <div class="card-hyper">
                    <h3>💎 EXCLUSIVE COLLECTION</h3>
                    <h1>{final_price:,.0f} €</h1>
                    <p style="color:#eee;">Bu araç sınıfı için özel koleksiyon değerleme algoritması kullanılmıştır.</p>
                </div>
            """, unsafe_allow_html=True)

        elif segment == "Super":
            # FERRARI TARZI KART
            st.markdown(f"""
                <div class="card-super">
                    <h3>🔥 SUPER SPORT VALUATION</h3>
                    <h1>{final_price:,.0f} €</h1>
                    <p style="color:#eee;">Yüksek performans segmenti yapay zeka modeli.</p>
                </div>
            """, unsafe_allow_html=True)

        else:
            # STANDART DROPNA KARTI
            st.markdown(f"""
                <div class="card-std">
                    <h3>TAHMİNİ PİYASA DEĞERİ</h3>
                    <h1>{final_price:,.0f} <span class="currency-symbol">€</span></h1>
                    <p style="color:#5b6d85; font-size:12px; margin-top:15px;">
                        * DropNA AI Algoritması Tarafından Hesaplanmıştır.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.balloons()

    elif error_message:
        st.error(f"Hata: {error_message}")
