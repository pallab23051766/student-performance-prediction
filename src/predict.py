import joblib
import pandas as pd


def predict_score(input_data, model_path="models/student_model.pkl"):
    model = joblib.load(model_path)

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    return round(prediction, 2)