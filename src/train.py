import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.feature_engineering import get_features_and_target


def train_model(data_path, model_path):
    df = pd.read_csv(data_path)

    X, y = get_features_and_target(df)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model_name = None
    best_pipeline = None
    best_mae = float("inf")
    best_r2 = -1

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{name} -> MAE: {mae:.2f}, R2: {r2:.2f}")

        if r2 > best_r2:
            best_model_name = name
            best_pipeline = pipeline
            best_mae = mae
            best_r2 = r2

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    print(f"\nBest Model: {best_model_name}")

    # Feature Importance Graph (only for Random Forest)
    if best_model_name == "Random Forest":
        model = best_pipeline.named_steps["model"]
        preprocessor = best_pipeline.named_steps["preprocessor"]

        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\nTop 10 Important Features:")
        print(importance_df.head(10))

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"].head(10), importance_df["Importance"].head(10))
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title("Top 10 Feature Importances")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    return best_model_name, best_mae, best_r2


if __name__ == "__main__":
    data_path = "data/processed/student_clean.csv"
    model_path = "models/student_model.pkl"

    best_model_name, mae, r2 = train_model(data_path, model_path)

    print("\nTraining completed.")
    print(f"Best Model: {best_model_name}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")