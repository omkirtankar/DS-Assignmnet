"""
Train a House Price Prediction Model
Dataset: California Housing (scikit-learn / Kaggle)
Algorithm: Gradient Boosting Regressor
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os

def train_and_save_model():
    print("=" * 60)
    print("  House Price Prediction - Model Training")
    print("=" * 60)

    # ── Load Dataset ──────────────────────────────────────────
    print("\n📦 Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    feature_names = housing.feature_names
    target_name = housing.target_names[0]

    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {feature_names}")
    print(f"   Target: {target_name} (Median House Value in $100k, converted to INR)")

    # ── Feature descriptions (for UI) ─────────────────────────
    feature_info = {
        "MedInc": {
            "label": "Median Income",
            "description": "Median income in block group (in ₹8,50,000s)",
            "min": float(df["MedInc"].min()),
            "max": float(df["MedInc"].max()),
            "mean": float(df["MedInc"].mean()),
            "step": 0.1
        },
        "HouseAge": {
            "label": "House Age",
            "description": "Median house age in block group (years)",
            "min": float(df["HouseAge"].min()),
            "max": float(df["HouseAge"].max()),
            "mean": float(df["HouseAge"].mean()),
            "step": 1
        },
        "AveRooms": {
            "label": "Average Rooms",
            "description": "Average number of rooms per household",
            "min": 1.0,
            "max": 15.0,
            "mean": float(df["AveRooms"].mean()),
            "step": 0.1
        },
        "AveBedrms": {
            "label": "Average Bedrooms",
            "description": "Average number of bedrooms per household",
            "min": 0.5,
            "max": 5.0,
            "mean": float(df["AveBedrms"].mean()),
            "step": 0.1
        },
        "Population": {
            "label": "Population",
            "description": "Block group population",
            "min": 3.0,
            "max": 10000.0,
            "mean": float(df["Population"].mean()),
            "step": 10
        },
        "AveOccup": {
            "label": "Avg Occupancy",
            "description": "Average number of household members",
            "min": 1.0,
            "max": 10.0,
            "mean": float(df["AveOccup"].mean()),
            "step": 0.1
        },
        "Latitude": {
            "label": "Latitude",
            "description": "Block group latitude",
            "min": float(df["Latitude"].min()),
            "max": float(df["Latitude"].max()),
            "mean": float(df["Latitude"].mean()),
            "step": 0.01
        },
        "Longitude": {
            "label": "Longitude",
            "description": "Block group longitude",
            "min": float(df["Longitude"].min()),
            "max": float(df["Longitude"].max()),
            "mean": float(df["Longitude"].mean()),
            "step": 0.01
        }
    }

    # ── Prepare Data ──────────────────────────────────────────
    X = df[feature_names]
    y = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📊 Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")

    # ── Scale Features ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train Model ───────────────────────────────────────────
    print("\n🚀 Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # USD to INR conversion rate
    USD_TO_INR = 85

    def format_inr(amount):
        """Format number in Indian numbering system (XX,XX,XXX)."""
        s = f"{int(amount)}"
        if len(s) <= 3:
            return s
        last3 = s[-3:]
        remaining = s[:-3]
        # Group remaining digits in pairs from right
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        return ','.join(groups) + ',' + last3

    mae_inr = mae * 100000 * USD_TO_INR
    rmse_inr = rmse * 100000 * USD_TO_INR

    print("\n📈 Model Performance (Test Set):")
    print(f"   MAE:  ₹{format_inr(mae_inr)}")
    print(f"   RMSE: ₹{format_inr(rmse_inr)}")
    print(f"   R²:   {r2:.4f} ({r2 * 100:.1f}%)")

    # ── Feature Importance ────────────────────────────────────
    importances = model.feature_importances_
    importance_dict = {}
    print("\n🔍 Feature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        importance_dict[name] = round(float(imp), 4)
        bar = "█" * int(imp * 50)
        print(f"   {name:12s} {imp:.4f} {bar}")

    # ── Save Artifacts ────────────────────────────────────────
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/house_price_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    print("\n💾 Model saved to model/house_price_model.pkl")
    print("💾 Scaler saved to model/scaler.pkl")

    # Save metadata
    metadata = {
        "feature_names": feature_names,
        "feature_info": feature_info,
        "feature_importance": importance_dict,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mae_inr": f"₹{format_inr(mae_inr)}",
            "rmse_inr": f"₹{format_inr(rmse_inr)}",
            "r2_percent": f"{r2 * 100:.1f}%"
        },
        "dataset": {
            "name": "California Housing",
            "source": "Kaggle / scikit-learn",
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "model": {
            "algorithm": "Gradient Boosting Regressor",
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.1
        }
    }

    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("💾 Metadata saved to model/metadata.json")

    print("\n" + "=" * 60)
    print("  ✅ Training Complete! Run 'python app.py' to start the app.")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_model()
