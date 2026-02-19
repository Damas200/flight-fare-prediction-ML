from data_loader import load_data
from preprocessing import clean_data
from feature_engineering import create_features
from modeling import train_models
from evaluation import evaluate_model
from utils import save_model


def main():

    print("Pipeline started...\n")

    df = load_data("data/Flight_Price_Dataset_of_Bangladesh.csv")
    df = clean_data(df)
    df = create_features(df)

    results, X_test, y_test = train_models(df, "Total Fare (BDT)")

    # Select best model based on CV
    best_model_name = max(
        results,
        key=lambda x: results[x]["cv_r2_mean"]
    )

    best_model = results[best_model_name]["model"]

    print(f"\nBest Model: {best_model_name}")

    evaluate_model(best_model, X_test, y_test)

    save_model(best_model)


if __name__ == "__main__":
    main()
