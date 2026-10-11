# ML Assignment 2 — Fault Detection

## Project Overview

This project implements machine learning models to detect electrical faults using current and voltage measurements. It compares Logistic Regression and Support Vector Machines (SVM) using two feature sets.

## Dataset

The dataset is stored in `data/detect_dataset.csv`.

The six original features are:
- `Ia`, `Ib`, `Ic` — three-phase current measurements
- `Va`, `Vb`, `Vc` — three-phase voltage measurements

The target variable is `fault`, where 0 represents no fault and 1 represents a fault.

## Feature Sets

- **Set R:** Six standardized original features.
- **Set Q:** The six original features plus their squared terms.

## Implemented Models

- Logistic Regression on Set R
- Logistic Regression on Set Q
- Linear SVM on Set R
- RBF SVM on Set R

The SVM models are selected using validation accuracy.

## Project Structure

- `data/` — Original dataset
- `src/load_data.py` — Loads the dataset
- `src/prepare.py` — Prepares the training, validation and test sets
- `src/eda.py` — Generates exploratory data analysis plots
- `src/logistic_regression.py` — Logistic Regression implementation
- `src/train_logistic.py` — Trains and evaluates Logistic Regression
- `src/metrics.py` — Calculates accuracy, confusion matrix and missed faults
- `src/train_svm.py` — Trains and compares SVM configurations
- `src/train_test.py` — Evaluates the four final models on the test set
- `results/` — Plots, CSV results, model files and learned parameters
- `app/` — Application folder

## Environment Setup

Python and the required libraries must be installed.

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If setting up the environment for the first time, install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn joblib
```

## How to Run

Run these commands from the project root directory:

```bash
python src/load_data.py
python src/prepare.py
python src/eda.py
python src/train_logistic.py
python src/train_svm.py
python src/train_test.py
```

Run the preparation and training scripts before the final test evaluation. The test dataset is used for final evaluation only.

## Results

The recorded test accuracies are:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression — Set R | 72.64% |
| Logistic Regression — Set Q | 97.88% |
| Linear SVM | 73.59% |
| RBF SVM | 99.58% |

The RBF SVM achieved the highest test accuracy and missed 6 actual fault samples in the recorded evaluation.

Detailed results are available in `results/test_results.csv`. Analysis and figures are available in the `results/` directory.

## Notes

- Run commands from the project root directory.
- The training and validation datasets are used for model selection.
- The test dataset is reserved for final evaluation.
## Frontend Screenshot

![Electrical Fault Detection App](results/frontend_screenshot.png)