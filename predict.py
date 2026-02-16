import joblib
import pandas as pd

MODEL_PATH = "models/price_model.joblib"

def predict_example():
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")

    # Example input (must match FEATURES order in train.py)
    example_house = pd.DataFrame([{
        "area": 1600,
        "bedrooms": 3,
        "bathrooms": 2
    }])

    prediction = model.predict(example_house)[0]
    print(f"🏠 Predicted house price: {prediction:,.2f}")

if __name__ == "__main__":
    predict_example()