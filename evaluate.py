"""Evaluate a face verification model on the eval1 dataset.

Usage:
    python evaluate.py <model_file>
"""

from argparse import ArgumentParser
import os

from joblib import load

IMAGE_HEIGHT = 62
IMAGE_WIDTH = 47
MODEL_SIZE_LIMIT_MB = 80


def normalize_pairs(x_test):
    """Apply the same per-face normalization used during training."""
    x_test = x_test / 255.0
    faces = x_test.reshape(-1, IMAGE_HEIGHT * IMAGE_WIDTH)
    faces = (faces - faces.mean(axis=1, keepdims=True)) / (faces.std(axis=1, keepdims=True) + 1e-8)
    return faces.reshape(x_test.shape[0], -1)


def evaluate(model_file):
    """Evaluate a saved model artifact against data/eval1.joblib."""
    print(f"Evaluating {model_file}")

    if os.path.getsize(model_file) > MODEL_SIZE_LIMIT_MB * 1024 * 1024:
        print(f"ERROR: Model file is larger than the allowed {MODEL_SIZE_LIMIT_MB} MB limit.")
        return

    model_data = load(model_file)

    if isinstance(model_data, dict):
        model = model_data["model"]
        scaler = model_data.get("scaler")
        pca = model_data.get("pca")
    else:
        model = model_data
        scaler = None
        pca = None

    eval_data = load("data/eval1.joblib")
    x_test = normalize_pairs(eval_data["data"])
    y_test = eval_data["target"]

    if scaler is not None:
        x_test = scaler.transform(x_test)
    if pca is not None:
        x_test = pca.transform(x_test)

    score = model.score(x_test, y_test)
    print(f"Score: {score * 100:.2f}%")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a trained face verification model")
    parser.add_argument("model_file", type=str, help="Path to the trained model file")
    args = parser.parse_args()
    evaluate(args.model_file)