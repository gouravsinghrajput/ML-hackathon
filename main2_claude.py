"""
Demand Prediction — High-Efficiency ML Pipeline
================================================
Model Stack: LightGBM + XGBoost + Random Forest → Ridge Meta-Learner
Target    : demand (continuous, range ≈ 0–1)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 60)
print("  DEMAND PREDICTION PIPELINE")
print("=" * 60)

train = pd.read_csv("/mnt/user-data/uploads/train.csv")
test  = pd.read_csv("/mnt/user-data/uploads/test.csv")

print(f"\n[Data] Train: {train.shape}  |  Test: {test.shape}")

# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # --- Timestamp → numeric hour + minute + cyclic features ---
    time_parts = df["timestamp"].str.split(":", expand=True).astype(float)
    df["hour"]   = time_parts[0]
    df["minute"] = time_parts[1]
    df["time_minutes"] = df["hour"] * 60 + df["minute"]  # 0–1435

    # Cyclic encoding (captures periodicity of the day)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]   / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]   / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    # Day cyclic (assuming weekly pattern; adjust period if different)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    # Is rush hour?
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Is midnight/late night?
    df["night"] = (df["hour"] <= 5).astype(int)

    # --- Geohash → spatial features ---
    # Geohash prefix clusters (coarser = broader region)
    df["geo_prefix_3"] = df["geohash"].str[:3]
    df["geo_prefix_4"] = df["geohash"].str[:4]
    df["geo_prefix_5"] = df["geohash"].str[:5]

    # Label-encode geohash prefixes
    for col in ["geo_prefix_3", "geo_prefix_4", "geo_prefix_5", "geohash"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # --- Categorical encoding ---
    # RoadType
    road_map = {"Residential": 0, "Street": 1, "Highway": 2}
    df["RoadType_enc"] = df["RoadType"].map(road_map).fillna(-1)

    # Weather severity score
    weather_map = {"Sunny": 0, "Cloudy": 1, "Foggy": 2, "Rainy": 3, "Snowy": 4}
    df["Weather_enc"] = df["Weather"].map(weather_map).fillna(-1)

    # LargeVehicles allowed?
    df["LargeVehicles_enc"] = (df["LargeVehicles"] == "Allowed").astype(int)

    # Landmarks present?
    df["Landmarks_enc"] = (df["Landmarks"] == "Yes").astype(int)

    # --- Temperature imputation + interaction ---
    temp_median = df["Temperature"].median()
    df["Temperature_missing"] = df["Temperature"].isna().astype(int)
    df["Temperature"] = df["Temperature"].fillna(temp_median)

    # Interaction: cold weather + night → likely lower demand
    df["cold_night"] = ((df["Temperature"] < 10) & (df["night"] == 1)).astype(int)

    # Highway + large vehicles → high-demand proxy
    df["highway_large"] = (
        (df["RoadType_enc"] == 2) & (df["LargeVehicles_enc"] == 1)
    ).astype(int)

    # Lanes × road type interaction
    df["lanes_road"] = df["NumberofLanes"] * (df["RoadType_enc"] + 1)

    return df


train_fe = engineer_features(train)
test_fe  = engineer_features(test)

# ─────────────────────────────────────────
# 3. DEFINE FEATURES
# ─────────────────────────────────────────
FEATURES = [
    # Temporal
    "hour", "minute", "time_minutes",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "day", "day_sin", "day_cos",
    "rush_hour", "night",
    # Spatial
    "geohash", "geo_prefix_3", "geo_prefix_4", "geo_prefix_5",
    # Road
    "RoadType_enc", "NumberofLanes", "LargeVehicles_enc",
    "Landmarks_enc", "lanes_road", "highway_large",
    # Weather
    "Weather_enc", "Temperature", "Temperature_missing", "cold_night",
]

TARGET = "demand"

X = train_fe[FEATURES]
y = train_fe[TARGET]
X_test = test_fe[FEATURES]

print(f"[Features] {len(FEATURES)} features used")
print(f"[Target]   mean={y.mean():.4f}, std={y.std():.4f}, max={y.max():.4f}\n")

# ─────────────────────────────────────────
# 4. BASE MODELS
# ─────────────────────────────────────────
lgbm_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

xgb_model = xgb.XGBRegressor(
    n_estimators=1200,
    learning_rate=0.04,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=5,
    max_features=0.6,
    random_state=42,
    n_jobs=-1,
)

# ─────────────────────────────────────────
# 5. STACKING ENSEMBLE
# ─────────────────────────────────────────
stack = StackingRegressor(
    estimators=[
        ("lgbm", lgbm_model),
        ("xgb",  xgb_model),
        ("rf",   rf_model),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1,
    passthrough=False,
)

# ─────────────────────────────────────────
# 6. CROSS-VALIDATION EVALUATION
# ─────────────────────────────────────────
print("─" * 60)
print("  Cross-Validation (5-Fold) on Individual Models")
print("─" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [("LightGBM", lgbm_model), ("XGBoost", xgb_model), ("RandomForest", rf_model)]:
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"  {name:<14} R² = {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────
# 7. TRAIN STACKED MODEL + HOLD-OUT EVAL
# ─────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

print("\n─" * 30)
print("  Training Stacked Ensemble …")
stack.fit(X_tr, y_tr)

y_pred_val = stack.predict(X_val)

r2  = r2_score(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\n  ✅ Validation Results (15% hold-out)")
print(f"     R²   = {r2:.4f}")
print(f"     MAE  = {mae:.4f}")
print(f"     RMSE = {rmse:.4f}")

# ─────────────────────────────────────────
# 8. RETRAIN ON FULL DATA + PREDICT TEST
# ─────────────────────────────────────────
print("\n  Retraining on full training data …")
stack.fit(X, y)

test_preds = stack.predict(X_test)
test_preds = np.clip(test_preds, 0, 1)  # demand is bounded [0,1]

# ─────────────────────────────────────────
# 9. SAVE PREDICTIONS
# ─────────────────────────────────────────
submission = test[["Index"]].copy()
submission["demand"] = test_preds
submission.to_csv("/mnt/user-data/outputs/predictions.csv", index=False)

print(f"\n  📁 Predictions saved → predictions.csv")
print(f"     Rows: {len(submission)}")
print(f"     Demand range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
print("=" * 60)

# ─────────────────────────────────────────
# 10. FEATURE IMPORTANCE (LightGBM)
# ─────────────────────────────────────────
lgbm_model.fit(X, y)
importances = pd.Series(lgbm_model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("\n  Top 15 Feature Importances (LightGBM)")
print("─" * 40)
for feat, imp in importances.head(15).items():
    bar = "█" * int(imp / importances.max() * 30)
    print(f"  {feat:<22} {bar} {imp:.0f}")
print("=" * 60)