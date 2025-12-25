import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# 1. LINEER MODEL ICIN OZEL HAZIRLIK (DATA CLEANING)
# =============================================================================
# Not: Önceki FE kodundaki adımların (car_age, log_mileage vb.) yapıldığını varsayıyoruz.

def prepare_for_linear_models(df):
    # Lineer modeller NaN sevmez, basitçe dolduralım
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Target'ı çıkaralım
    if "price_log" in num_cols: num_cols.remove("price_log")
    if "price" in num_cols: num_cols.remove("price")

    # Pipeline: Sayısal veriler için Ölçeklendirme, Kategorikler için OneHot
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Lineer modeller için kritik!
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    return preprocessor


# =============================================================================
# 2. MODEL DENEYLERI VE PERFORMANS KARŞILAŞTIRMA
# =============================================================================
# X ve y'nin hazır olduğunu varsayıyoruz (price_log hedef değişken)
X = df.drop(columns=["price", "price_log"], errors='ignore')
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer Model Listesi
linear_models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge (L2)', Ridge(alpha=1.0)),
    ('Lasso (L1)', Lasso(alpha=0.01))
]

results = []

print(f"{'Model':<20} | {'Train RMSE':<12} | {'Test RMSE':<12} | {'R2 Score':<10}")
print("-" * 65)

preprocessor = prepare_for_linear_models(X)

for name, model in linear_models:
    # Pipeline birleştirme: Preprocess + Model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', model)])

    # Eğitim
    model_pipeline.fit(X_train, y_train)

    # Tahmin
    train_preds = model_pipeline.fit(X_train, y_train).predict(X_train)
    test_preds = model_pipeline.predict(X_test)

    # Metrikler
    rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)

    results.append((name, rmse_test, r2))
    print(f"{name:<20} | {rmse_train:<12.4f} | {rmse_test:<12.4f} | {r2:<10.4f}")