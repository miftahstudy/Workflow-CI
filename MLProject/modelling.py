import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set MLflow Tracking URI lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Loan Approval Prediction")


def load_data():
    data_path = "loan_approval_preprocessing/loan_approval_preprocessed.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    return X, y


def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Basic requirement: menggunakan autolog MLflow
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Training selesai.")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    train_model()