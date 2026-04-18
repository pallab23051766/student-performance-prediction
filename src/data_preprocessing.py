import pandas as pd
import os


def load_data(mat_path, por_path):
    df_mat = pd.read_csv(mat_path, sep=';')
    df_por = pd.read_csv(por_path, sep=';')
    return df_mat, df_por


def merge_data(df_mat, df_por):
    df_mat["course"] = "math"
    df_por["course"] = "portuguese"

    df = pd.concat([df_mat, df_por], ignore_index=True)
    df.drop_duplicates(inplace=True)
    return df


def clean_data(df):
    df = df.copy()

    # remove missing values if any
    df.dropna(inplace=True)

    # optional: remove spaces from column names
    df.columns = [col.strip() for col in df.columns]

    return df


def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def run_preprocessing(mat_path, por_path, output_path):
    df_mat, df_por = load_data(mat_path, por_path)
    df = merge_data(df_mat, df_por)
    df = clean_data(df)
    save_processed_data(df, output_path)
    return df


if __name__ == "__main__":
    mat_path = "data/raw/student-mat.csv"
    por_path = "data/raw/student-por.csv"
    output_path = "data/processed/student_clean.csv"

    df = run_preprocessing(mat_path, por_path, output_path)
    print("Preprocessing completed.")
    print("Processed shape:", df.shape)