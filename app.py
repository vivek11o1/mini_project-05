import joblib
import pandas as pd
import gradio as gr
import os

MODEL_PATH = "models/price_model.joblib"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please run train.py first.")
    return joblib.load(MODEL_PATH)

model = load_model()

def predict_price(area, bedrooms, bathrooms):
    data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms
    }])
    prediction = model.predict(data)[0]
    return f"🏠 Estimated House Price: {prediction:,.2f}"

def launch_ui():
    demo = gr.Interface(
        fn=predict_price,
        inputs=[
            gr.Number(label="Area (sq ft)", value=1200),
            gr.Number(label="Bedrooms", value=3),
            gr.Number(label="Bathrooms", value=2),
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="🏡 PriceNest — House Price Predictor",
        description="Enter house details to get an estimated price using a trained ML model."
    )
    return demo.launch()

