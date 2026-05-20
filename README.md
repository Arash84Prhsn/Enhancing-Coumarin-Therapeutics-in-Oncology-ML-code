# ML Implementation for "Machine Learning-Guided Dose-Time Optimization and Experimental Validation Enhance Coumarin Therapeutics in Oncology"

> **Note:** This repository is currently intended primarily for peer reviewers and collaborators during the review process. The repository and/or dataset may be made private or removed after publication acceptance.

This repository contains:

- All machine learning implementations used in the study
- Python scripts used to generate several figures from the manuscript
- The preprocessing pipeline used in the study
- Trained models and saved training results for reproducibility

The primary implementation is available in the `main.ipynb` notebook, while standalone Python scripts corresponding to notebook cells can be found in the `src/` directory.

---

# Repository Structure

```text
.
├── main.ipynb
├── requirements.txt
├── Total_Data.csv
├── Processed_data.csv
├── src/
│   ├── Preprocessing.py
│   ├── ...
│
├── TrainingResults/
│   ├── Model_1/
│   ├── Model_2/
│   ├── ...
│
├── Figures/
│   └── ...
```

---

# System Requirements

- Python **3.13.7**
- Git (recommended)

The following Python packages are required:

| Package | Version |
|---|---|
| scikit-learn | 1.7.2 |
| xgboost | 3.1.1 |
| pandas | 2.3.3 |
| numpy | 2.3.4 |
| joblib | 1.5.2 |
| matplotlib | 3.10.7 |
| graphviz | 0.21 |

The following standard library modules are also used:

- `os`
- `sys`

---

# 1. Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/poorhasaniarash2/Enhancing-Coumarin-Therapeutics-in-Oncology-ML.git
```

Move into the repository directory:

```bash
cd Enhancing-Coumarin-Therapeutics-in-Oncology-ML
```

If downloading manually instead of using Git, extract the ZIP file and open the project folder in your preferred IDE or editor.

---

# 2. Create a Virtual Environment (Recommended)

## Linux / macOS

Create the virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

## Windows (PowerShell)

Create the virtual environment:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
venv\Scripts\Activate.ps1
```

## Windows (CMD)

```cmd
venv\Scripts\activate.bat
```

---

# 3. Install Dependencies

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the packages manually:

```bash
pip install \
scikit-learn==1.7.2 \
xgboost==3.1.1 \
pandas==2.3.3 \
numpy==2.3.4 \
joblib==1.5.2 \
matplotlib==3.10.7 \
graphviz==0.21
```

---

# Data Files

The repository includes two datasets:

| File | Description |
|---|---|
| `Total_Data.csv` | Raw experimental dataset |
| `Processed_data.csv` | Dataset after preprocessing |

---

# Preprocessing Pipeline

The preprocessing pipeline is implemented in:

```text
src/Preprocessing.py
```

Run the preprocessing script using:

```bash
python src/Preprocessing.py
```

This script generates:

```text
Processed_data.csv
```

The preprocessing pipeline includes:

- GMM-based reliability filtering
- Encoding of the `CancerType` feature
- Encoding of the `Coumarin` feature
- Application of time constraints for Auraptene and other coumarins

---

# GMM Reliability Filtering

Because the dataset is relatively small and distributed across 18 cancer types, a Gaussian Mixture Model (GMM) filtering stage was introduced to improve data reliability.

The preprocessing script performs the following steps:

1. Creates a dataframe containing:
   - `Cancer Type`
   - `Sample Count`

2. Fits a Gaussian Mixture Model with:

```python
n_components = 2
random_state = 42
```

3. Calculates the inclusion threshold as the average of the two Gaussian component means.

4. Assigns a reliability label to each cancer type.

5. Excludes cancer types deemed unreliable from the final processed dataset.

---

# Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
main.ipynb
```

Run the notebook cells sequentially from top to bottom.

---

# Running the Python Scripts

All notebook cells are also available as standalone scripts inside:

```text
src/
```

Run scripts using:

```bash
python src/<SCRIPT_NAME>.py
```

Example:

```bash
python src/RandomForest.py
```

---

# Model Training and Grid Search

Each model has its own script and notebook section.

When running a model script, the program will prompt:

```text
Perform grid search? (yes/no)
```

## If you answer `yes`

The script will:

1. Perform hyperparameter grid search
2. Save the best hyperparameters
3. Store results in the corresponding `TrainingResults/` subdirectory
4. Exit after grid search completes

## If you answer `no`

The script will:

1. Load the saved best hyperparameters
2. Train the model on the processed dataset
3. Perform cross-validation
4. Print evaluation metrics and predictions to the terminal

---

# Training Results

The `TrainingResults/` directory contains subdirectories for each trained model.

Each model directory contains:

- Best hyperparameters discovered during grid search
- Cross-validation scores
- Saved trained model objects (`.joblib`)

Example:

```text
TrainingResults/
└── RandomForest/
    ├── best_params.txt
    ├── cv_scores.txt
    └── model.joblib
```

---

# Full Reproducibility Guide

To fully reproduce all results from scratch:

## Step 1 — Clone the Repository

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
```

## Step 2 — Create and Activate Virtual Environment

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Delete Existing Generated Files

Delete:

```text
Processed_data.csv
TrainingResults/
```

This ensures all outputs are regenerated from scratch.

## Step 5 — Run Preprocessing

```bash
python src/Preprocessing.py
```

## Step 6 — Run Model Scripts

Run each model script in the same order as they appear in `main.ipynb`.

For the first run of each model:

- Respond with:

```text
yes
```

to perform grid search.

Run the script again afterward and respond with:

```text
no
```

to perform training and evaluation using the discovered hyperparameters.

## Step 7 — Generate Figures

Run the figure-generation scripts to reproduce manuscript figures:

- Figure 2
- Figure 3-A
- Figure 3-B
- Figure 5

---

# Expected Outcome

After completing all steps:

- `Processed_data.csv` will be regenerated
- A new `TrainingResults/` directory will be created
- Trained models and evaluation metrics will match the repository outputs
- Figures from the manuscript will be regenerated successfully

---

# Recommended Development Environment

The repository was primarily developed and tested using:

- Visual Studio Code
- Python 3.13.7

Other IDEs/editors should also work correctly.

---