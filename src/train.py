"""Train a feedforward neural network for Rossmann store sales forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

FEATURES = [
    "Open",
    "Promo",
    "Promo2Open",
    "DayOfWeek",
    "CompetitionDistance",
    "CompetitionOpen",
    "Month",
    "DayOfYear",
]


def build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mse"])
    return model


def train_and_predict(
    processed_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.3,
    random_state: int = 42,
) -> Path:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train_preprocessed.csv"
    test_path = processed_dir / "test_preprocessed.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python -m src.preprocess` before training."
            )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x = train_data[FEATURES]
    y = train_data["Sales"]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=validation_split, random_state=random_state
    )

    feature_scaler = StandardScaler()
    x_train_scaled = feature_scaler.fit_transform(x_train)
    x_val_scaled = feature_scaler.transform(x_val)
    x_test_scaled = feature_scaler.transform(test_data[FEATURES])

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))

    model = build_model(x_train_scaled.shape[1])
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        x_train_scaled,
        y_train_scaled,
        validation_data=(x_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    val_loss, val_mse = model.evaluate(x_val_scaled, y_val_scaled, verbose=0)
    y_val_pred = target_scaler.inverse_transform(model.predict(x_val_scaled, verbose=0))
    rmse = float(np.sqrt(np.mean((y_val.values - y_val_pred.ravel()) ** 2)))
    print(f"Validation MSE (scaled): {val_mse:.6f}")
    print(f"Validation RMSE (sales): {rmse:,.2f}")

    test_predictions = target_scaler.inverse_transform(model.predict(x_test_scaled, verbose=0))
    submission = pd.DataFrame({"Id": test_data["Id"], "Sales": test_predictions.ravel()})
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    return submission_path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Rossmann sales FNN and export predictions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory with preprocessed CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs",
        help="Directory for submission output",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_predict(args.data_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
