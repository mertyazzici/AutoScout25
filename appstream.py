import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
import os  # <--- BU EKSİKTİ, EKLENDİ.

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
# 2. PREMIUM CSS TASARIMI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stApp { background-color: #0b1426; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #080f1f; border-right: 1px solid #cfa86050; }
    .sidebar-brand { text-align: center; font-size: 2rem; font-weight: 800; color: #cfa860; margin-bottom: 10px; }
    .premium-header { background: linear-gradient(90deg, #162447 0%, #1f4068 100%); padding: 2rem; border-radius: 12px; text-align: center; border-bottom: 3px solid #cfa860; margin-bottom: 2rem; }
    .header-title { font-size: 2.8rem; font-weight: 700; color: #ffffff; margin: 0; }
    .stButton>button { background: linear-gradient(135deg, #cfa860 0%, #b08d55 100%); color: #0b1426; border: none; height: 3.5em; font-weight: 700; border-radius: 8px; width: 100%; }

    /* Kart Tasarımları */
    .card-hyper { background: linear-gradient(135deg, #141E30, #243B55); color: #FFD700; padding: 40px; border-radius: 15px; text-align: center; border: 2px solid #FFD700; margin-top: 20px; }
    .card-super { background: linear-gradient(135deg, #8E0E00, #1F1C18); color: white; padding: 40px; border-radius: 15px; text-align: center; border: 1px solid #ff4d4d; margin-top: 20px; }
    .card-std { background-color: #162447; border: 1px solid #cfa860; border-radius: 12px; padding: 40px; text-align: center; margin-top: 20px; }
    .card-std h1, .card-hyper h1, .card-super h1 { font-size: 4rem; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. DOSYALARI VE MODELLERİ YÜKLEME
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
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">DropNA</div>', unsafe_allow_html=True)

    # Görsel hatasını engelleyen güvenli kontrol
    img_path = "dropna_team.jpeg"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning(f"Görsel bulunamadı! ({img_path})")

    st.write("---")
    st.info("Nasıl Kullanılır?\n1. Marka/Model seçin.\n2. Özellikleri girin.\n3. Değerleme Yap butonuna basın.")

# -----------------------------------------------------------------------------
# 5. GİRİŞLER VE HESAPLAMA
# -----------------------------------------------------------------------------
st.markdown('<div class="premium-header"><h1 class="header-title">🏎️ AUTOSCOUT25</h1></div>', unsafe_allow_html=True)

all_makes = sorted(metadata["make_model_map"].keys())
selected_make = st.selectbox("Marka Seçiniz", all_makes)

# Segment Belirleme
segment = "Standard"
if selected_make in metadata.get("hyper_makes", []):
    segment = "Hyper"
elif selected_make in metadata.get("super_makes", []):
    segment = "Super"

# Model Seçimi
models = sorted(metadata["make_model_map"][selected_make])
selected_model = st.selectbox("Model Seçiniz", models)

col1, col2 = st.columns(2)
with col1:
    km = st.number_input("Kilometre", 0, 1000000, 50000)
    year = st.number_input("Yıl", 1990, 2025, 2020)
    hp = st.number_input("Motor Gücü (kW)", 0, 1500, 100)
    gear_count = st.slider("Vites Sayısı", 1, 10, 6)

with col2:
    fuel = st.selectbox("Yakıt", sorted([str(x) for x in metadata["cat_options"]["fuel_category"]]))
    trans = st.selectbox("Vites Tipi", sorted([str(x) for x in metadata["cat_options"]["transmission"]]))
    body = st.selectbox("Kasa Tipi", sorted([str(x) for x in metadata["cat_options"]["body_type"]]))
    color = st.selectbox("Renk", sorted([str(x) for x in metadata["cat_options"]["body_color"]]))
    uph = st.selectbox("Döşeme", sorted([str(x) for x in metadata["cat_options"]["upholstery"]]))

if st.button("DEĞERLEME YAP ➤"):
    # Veri Hazırlama
    input_data = {
        "make": selected_make, "model": selected_model, "production_year": year,
        "mileage_km_raw": km, "power_kw": hp, "gears": gear_count,
        "fuel_category": fuel, "transmission": trans, "body_type": body,
        "body_color": color, "upholstery": uph,
        # Varsayılanlar
        "fuel_cons_comb_l100_km": 5.0, "nr_seats": 5, "nr_doors": 5,
        "is_used": "Yes", "seller_is_dealer": "Yes", "nr_prev_owners": 1,
        "electric_range_km": 0, "electric_range_city_km": 0
    }

    price = 0
    calculated = False

    if segment == "Hyper":
        # Basit Hyper Kuralı
        base = 2_500_000
        age_dep = (2026 - year) * 0.02 * base
        km_dep = km * 10
        price = max(base * 0.7, base - age_dep - km_dep)
        calculated = True
    else:
        # Model Tahmini
        try:
            active_model = model_super if (segment == "Super" and model_super) else model_std
            cols = metadata.get("columns_order_super" if segment == "Super" else "columns_order_std",
                                metadata.get("columns_order", []))

            df = pd.DataFrame([input_data]).reindex(columns=cols)
            # Tip dönüşümü
            for c in df.columns:
                if c in metadata["cat_cols"]:
                    df[c] = df[c].astype(str)
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            log_price = active_model.predict(df)[0]
            price = np.expm1(log_price)
            calculated = True
        except Exception as e:
            st.error(f"Hesaplama hatası: {e}")

    if calculated:
        st.markdown("---")
        card_class = "card-hyper" if segment == "Hyper" else ("card-super" if segment == "Super" else "card-std")
        st.markdown(f"""
            <div class="{card_class}">
                <h3>TAHMİNİ FİYAT</h3>
                <h1>{price:,.0f} €</h1>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()