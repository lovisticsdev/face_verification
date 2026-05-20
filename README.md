# Face Verification

A Python machine learning project for binary face-pair verification. The pipeline normalizes face images, augments positive training pairs, reduces dimensionality with PCA, and trains an RBF-kernel SVM for same-person versus different-person classification.

## Contents

```text
Face Verification/
|-- train.py                 # Training, augmentation, model selection, and plot generation
|-- evaluate.py              # Evaluation script for the saved model
|-- model.joblib             # Saved model artifact: SVM, scaler, and PCA
|-- confusion_matrix.png     # Training holdout confusion matrix
|-- validation_scores.png    # PCA validation comparison plot
|-- data/
|   `-- eval1.joblib         # Local evaluation dataset, ignored by Git
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Requirements

The saved model was serialized with scikit-learn 1.5.1. For the cleanest reproduction, use Python 3.10-3.12 with the dependencies in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Place the training dataset at `train.joblib`, then run:

```powershell
python train.py train.joblib model.joblib
```

Training performs:

- per-face pixel normalization
- positive-pair augmentation using small rotations and blur
- PCA component selection
- SVM hyperparameter search with balanced accuracy
- final model export to `model.joblib`
- generation of `validation_scores.png` and `confusion_matrix.png`

## Evaluation

Place the evaluation dataset at `data/eval1.joblib`, then run:

```powershell
python evaluate.py model.joblib
```

The evaluator loads the saved SVM, scaler, and PCA objects from `model.joblib` and applies the same normalization path used during training.

## Git Notes

Large local datasets and temporary serialized artifacts are ignored by Git. The trained `model.joblib` is intentionally kept as the main deliverable artifact, while `train.joblib` and `data/*.joblib` remain local inputs.