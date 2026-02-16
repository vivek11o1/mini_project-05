🏡 PriceNest — House Price Prediction with Machine Learning

PriceNest is an end-to-end Machine Learning project that predicts house prices based on key features such as area, number of bedrooms, and bathrooms.
The project includes data preprocessing, model training, evaluation, CLI predictions, and an interactive Gradio web interface for real-time predictions.



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
