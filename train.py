import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_data, clean_data


def train_model(data_path="data/housing.csv", model_path="models/price_model.joblib"):
    print("🚀 Starting training...")

    # Check if dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {data_path}")

    # Load & clean data
    df = load_data(data_path)
    print("✅ Data loaded")

    df = clean_data(df)
    print("✅ Data cleaned")

    print("📊 Columns:", df.columns.tolist())

    # ---- Feature selection ----
    # ⚠️ Adjust these column names to match your dataset
    FEATURES = ["area", "bedrooms", "bathrooms"]  # change if needed
    TARGET = "price"

    # Validate columns
    missing_features = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_features:
        raise ValueError(f"❌ Missing columns in dataset: {missing_features}")

    X = df[FEATURES]
    y = df[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("🔀 Data split into train/test")

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("🧠 Model trained")

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Evaluation Metrics")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    # Print coefficients for interpretation
    print("\n🧮 Model Coefficients:")
    for feature, coef in zip(FEATURES, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    return model, rmse, r2


if __name__ == "__main__":
    train_model()