from src.data_preprocessing import run_preprocessing
from src.train import train_model


def run_pipeline():
    mat_path = "data/raw/student-mat.csv"
    por_path = "data/raw/student-por.csv"
    processed_path = "data/processed/student_clean.csv"
    model_path = "models/student_model.pkl"

    print("Running preprocessing...")
    run_preprocessing(mat_path, por_path, processed_path)

    print("\nTraining advanced model pipeline...")
    model_name, mae, r2 = train_model(processed_path, model_path)

    print("\nPipeline completed.")
    print(f"Best Model: {model_name}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")


if __name__ == "__main__":
    run_pipeline()