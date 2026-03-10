import pandas as pd
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from heart_utils import setup_logger, ensure_dir

logger = setup_logger('DataPrep')


def preprocess_data(raw_data_path, processed_data_path, model_dir):
    logger.info(f"Loading raw dataset from: {raw_data_path}")

    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {raw_data_path}")
        return

    # 1. Cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # 2. Separation
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    ensure_dir(model_dir)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"StandardScaler successfully saved to {scaler_path}")

    ensure_dir(processed_data_path)
    X_train_scaled.to_csv(os.path.join(processed_data_path, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_data_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_data_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_data_path, "y_test.csv"), index=False)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PATH = os.path.join(BASE_DIR, "heart.csv")
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    preprocess_data(RAW_PATH, PROCESSED_PATH, MODEL_DIR)