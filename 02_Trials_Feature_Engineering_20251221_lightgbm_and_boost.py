import pandas as pd
import numpy as np
import ast
import unicodedata
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# =============================================================================
# 1. CONFIG & HELPER FUNCTIONS
# =============================================================================
INPUT_PATH = "auto_processed_after_drop.csv"
OUTPUT_PATH = "model_ready_final.csv"
CURRENT_YEAR = 2026

def clean_col_name(text):
    """Standardizes column names for models like LightGBM."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("utf-8")
    text = text.lower().strip()
    text = re.sub(r'[\[\]<>\s\-/.]+', '_', text)
    text = re.sub(r'[\"\']', '', text)
    return text

def parse_list(x):
    if pd.isna(x): return []
    if isinstance(x, list): return x
    try:
        return ast.literal_eval(x)
    except:
        return []

# =============================================================================
# 2. DATA LOADING & PRE-CLEANING
# =============================================================================
df = pd.read_csv(INPUT_PATH)

# Drop redundant or non-predictive columns early
DROP_INITIAL = [
    "ratings_average", "new_is_rated", "model_version_en",
    "offer_type", "country_code", "model_key", "make_model"
]
df.drop(columns=[c for c in DROP_INITIAL if c in df.columns], inplace=True)
df.columns = [clean_col_name(c) for c in df.columns]

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
df["car_age"] = CURRENT_YEAR - df["production_year"]
df["km_per_year"] = df["mileage_km_raw"] / df["car_age"].replace(0, 1)
df["log_mileage"] = np.log1p(df["mileage_km_raw"])
df["price_log"] = np.log1p(df["price"])

df['is_premium'] = df['make'].str.lower().isin(
    ['audi', 'bmw', 'mercedes-benz', 'porsche', 'land rover', 'volvo']).astype(int)

df['power_group_le'] = pd.cut(df['power_kw'], bins=[-1, 66, 110, 200, float("inf")], labels=[0, 1, 2, 3]).astype(int)
df['mileage_cat_le'] = pd.cut(df['mileage_km_raw'], bins=[-1, 30000, 100000, 200000, float("inf")], labels=[0, 1, 2, 3]).astype(int)
df['inconsistency_flag'] = ((df['mileage_km_raw'] < 1000) & (df['car_age'] > 3)).astype(int)

# =============================================================================
# 4. EQUIPMENT FEATURES (LIST TO DUMMIES)
# =============================================================================
EQUIP_COLS = ['equipment_comfort', 'equipment_entertainment', 'equipment_extra', 'equipment_safety']

for col in EQUIP_COLS:
    if col in df.columns:
        df[col] = df[col].apply(parse_list)
        df[f"{col}_count"] = df[col].apply(len)
        all_items = [item for row in df[col] for item in row]
        top_10 = [x[0] for x in Counter(all_items).most_common(10)]
        for feat in top_10:
            df[f"{col}_{clean_col_name(feat)}"] = df[col].apply(lambda x: 1 if feat in x else 0)
        df.drop(columns=[col], inplace=True)

# =============================================================================
# 5. CATEGORICAL ENCODING & CLEANING
# =============================================================================
# Handle specific columns from the error log
df["drive_train"] = df["drive_train"].fillna("Unknown")
df["fuel_category"] = df["fuel_category"].fillna("Unknown")

# Body Type & Color
commercial = ["Transporter", "Panel van", "Van-high roof", "Flatbed van", "Box"]
df["body_type_grp"] = df["body_type"].apply(lambda x: "Commercial" if x in commercial else x)
color_map = {"Gray": "Grey", "Anthracite": "Grey", "Titanium": "Grey", "Burgundy": "Red"}
df["body_color"] = df["body_color"].replace(color_map)
df["color_neutral"] = df["body_color"].isin(["Black", "Grey", "White", "Silver"]).astype(int)

# Transmission & Fuel
df["is_automatic"] = (df["transmission"] == "Automatic").astype(int)
fuel_map = {"Electric/Gasoline": "Hybrid", "Electric/Diesel": "Hybrid", "LPG": "Alternative", "CNG": "Alternative"}
df["fuel_type_grp"] = df["fuel_category_filled"].replace(fuel_map)

# Upholstery, Euro6, Paint
uph_map = {"alcantara": "Alcantara", "Leather": "Full leather", "Cloth": "Cloth"}
df["upholstery_grp"] = df["upholstery"].replace(uph_map)
df["euro6_group"] = df["euro6_group"].map({"Euro 6": 1}).fillna(0).astype(int)
df["paint_type_metallic"] = df["paint_type"].map({"Metallic": 1}).fillna(0).astype(int)

# =============================================================================
# 6. FINAL DUMMIES & CLEANUP
# =============================================================================
# Group rare models
model_counts = df["model"].value_counts()
rare_models = model_counts[model_counts < 10].index
df["model"] = df["model"].replace(rare_models, "Other_Model")

# Identify columns to turn into dummies
DUMMY_COLS = [
    "make", "model", "fuel_type_grp", "upholstery_grp",
    "body_type_grp", "drive_train", "fuel_category"
]
df = pd.get_dummies(df, columns=[c for c in DUMMY_COLS if c in df.columns], drop_first=True)

# Drop any remaining text columns that aren't useful or were replaced
FINAL_DROP = [
    "price", "mileage_km_raw", "power_kw", "segment", "body_type",
    "body_color", "transmission", "fuel_category_filled", "upholstery",
    "paint_type", "nr_prev_owners", "envir_standard",
    "mileage_fill_reason", "production_year_source"
]
df.drop(columns=[c for c in FINAL_DROP if c in df.columns], inplace=True)

# Final check: Drop any columns that are still "object" type
object_cols = df.select_dtypes(include=['object']).columns
df.drop(columns=object_cols, inplace=True)

# =============================================================================
# 7. MODELING PREPARATION (FIXED)
# =============================================================================
X = df.drop(columns=["price_log"])
y = df["price_log"]

# 1. Remove JSON-breaking characters (":", "{", "}", "[", "]", ",", "\"")
X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)) for col in X.columns]

# 2. Robust Deduplication (Manual fix for your KeyError)
new_cols = []
column_counts = {}

for col in X.columns:
    if col in column_counts:
        column_counts[col] += 1
        new_cols.append(f"{col}_{column_counts[col]}")
    else:
        column_counts[col] = 0
        new_cols.append(col)

X.columns = new_cols

# 3. Final Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Final feature count: {X.shape[1]}")

# =============================================================================
# 8. MODEL EVALUATION
# =============================================================================
def evaluate_models(X_train, X_test, y_train, y_test):
    models = [
        ('LightGBM', LGBMRegressor(random_state=42, verbose=-1)),
        ('XGBoost', XGBRegressor(random_state=42)),
        ('CatBoost', CatBoostRegressor(random_state=42, verbose=0, iterations=500))
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # Convert log back to actual price for more intuitive RMSE
        rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(preds)))
        r2 = r2_score(y_test, preds)
        print(f"{name} -> RMSE (Price): {rmse:.2f}, R2: {r2:.4f}")

evaluate_models(X_train, X_test, y_train, y_test)

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSuccess. Dataset saved as '{OUTPUT_PATH}'.")