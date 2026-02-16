from data_loader import load_data, clean_data, explore_data
from train import train_model
from predict import predict_example
from app import launch_ui

def main():
    print("🔧 Step 1: Loading & exploring data...")
    df = load_data("data/housing.csv")
    df = clean_data(df)
    explore_data(df)

    print("\n🤖 Step 2: Training model...")
    train_model()

    print("\n🔮 Step 3: Running example prediction...")
    predict_example()

    print("\n🌐 Step 4: Launching UI (once)...")
    server = launch_ui()
    print("✅ UI launched. Check the terminal for the local URL.")

if __name__ == "__main__":
    main()