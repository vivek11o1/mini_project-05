🏡 PriceNest — House Price Prediction with Machine Learning

PriceNest is an end-to-end Machine Learning project that predicts house prices based on key features such as area, number of bedrooms, and bathrooms.
The project includes data preprocessing, model training, evaluation, CLI predictions, and an interactive Gradio web interface for real-time predictions.

## 🚀 FastAPI Service (Swagger UI)

A REST API is available alongside the original Gradio UI. The Gradio `app.py` is unchanged; the API lives in `api.py`.

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Open **http://localhost:8000/docs** for Swagger UI. The model is not committed to Git (`models/*.joblib` is gitignored), so `api_helper.py` trains it on first start from the bundled `melb_data.csv` (mapping `BuildingArea` → `area`, `Bedroom2` → `bedrooms`, `Bathroom` → `bathrooms`, `Price` → `price`).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | API information |
| GET | `/health` | Health check and feature names |
| POST | `/predict` | Price one house |
| POST | `/predict/batch` | Price several houses |
| POST | `/predict/csv` | Upload a CSV with `area, bedrooms, bathrooms` columns |
| GET | `/docs` | Swagger UI |

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"area": 1500, "bedrooms": 3, "bathrooms": 2}'
```

Example response:

```json
{"area": 1500.0, "bedrooms": 3.0, "bathrooms": 2.0, "predicted_price": 1234567.89}
```

Smoke test (with the server running): `python test_api.py`.

### Docker / deployment

```bash
docker build -t pricenest .
docker run -p 8000:8000 pricenest
```

Open **http://localhost:8000/docs**. The container binds to `0.0.0.0` and reads `PORT` (Render injects `PORT=10000`), falling back to `8000` locally.

On **Render**, create a **Web Service** with:

- **Runtime:** Docker (included `Dockerfile` uses Python 3.12)
- **Health Check Path:** `/health` (initial delay ~15s)
- No build/start commands needed.

If using Render's **Python** runtime instead (dashboard shows `python3`), set:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`

✨ Features
	•	📊 Load and clean housing dataset
	•	🧠 Train a Linear Regression model
	•	📈 Evaluate performance using RMSE and R²
	•	🔮 Make predictions from the command line
	•	🌐 Interactive Gradio web UI for live predictions
	•	🗂️ Clean, modular project structure



🗂️ Project Structure

PriceNest/
├── data/
│   └── housing.csv
├── models/
│   └── price_model.joblib        
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── app.py                      
├── main.py                       
├── requirements.txt
├── .gitignore
└── README.md



⚙️ Setup

1️⃣ Clone the repository

git clone https://github.com/Hassanmahmood4/PriceNest.git
cd PriceNest

2️⃣ Create virtual environment (optional but recommended)

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

3️⃣ Install dependencies

pip install -r requirements.txt



▶️ How to Run

🔧 Train the model

python src/train.py

This will train the model and save it to:

models/price_model.joblib



🔮 Run CLI Prediction

python src/predict.py



🌐 Launch Gradio Web UI

python app.py

Open the local URL shown in the terminal to access the UI.


🚀 Run Full Pipeline (Single Command)

python main.py

This will:
	•	Load & explore data
	•	Train the model
	•	Run a sample prediction
	•	Launch the web UI



📊 Model Details
	•	Algorithm: Linear Regression
	•	Evaluation Metrics:
	•	RMSE (Root Mean Squared Error)
	•	R² Score
	•	Features: Area, Bedrooms, Bathrooms (can be adapted for larger datasets)


📁 Dataset

You can use:
	•	A small custom CSV (data/housing.csv)
	•	Or a larger Kaggle dataset such as:
	•	California Housing Prices
	•	House Prices: Advanced Regression Techniques

Update features.py accordingly if you change the dataset.



🛠️ Tech Stack
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	Joblib
	•	Gradi

🌱 Future Improvements
	•	Add more advanced models (Random Forest, XGBoost)
	•	Feature scaling and feature engineering
	•	Model comparison dashboard
	•	Deployment to Hugging Face Spaces


👨‍💻 Author

Hassan Mahmood
GitHub: https://github.com/Hassanmahmood4
