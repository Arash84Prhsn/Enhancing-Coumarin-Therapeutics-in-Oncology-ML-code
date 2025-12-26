import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from joblib import load

DATA_PATH = 'Total_Data.csv'


mainData = pd.read_csv(DATA_PATH);

mainData = mainData[['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time', 'Viability']].dropna();

allowed_times_NoAuraptene = [24, 48, 72]
allowed_times_ForAuraptene = [24, 48, 72, 96]

coumarins = ['Auraptene', 'Esculetin', 'Galbanic Acid', 'Umbelliprenin'];

# ====<Pre-Processing and GMM filtering>====

#First filter the main data and save it in data.
cancer_counts = mainData['Cancer Type'].value_counts().to_dict()
count_df = pd.DataFrame(list(cancer_counts.items()), columns=['Cancer Type', 'Sample Count'])

#Initialize the gmm model
gmm = GaussianMixture(n_components=2, random_state=42);
gmm.fit(count_df[['Sample Count']])
threshold = np.mean(gmm.means_.flatten())

count_df['Reliability'] = count_df['Sample Count'].apply(
    lambda x: 'Reliable' if x >= threshold else 'Unreliable'
)

reliable_cancers = count_df[count_df['Reliability'] == 'Reliable']['Cancer Type'].tolist()
reliable_data = mainData[mainData['Cancer Type'].isin(reliable_cancers)].copy()

data = reliable_data;
CancerType_Encoder = LabelEncoder();
CoumarinType_Encoder = LabelEncoder();

data['Coumarin Type'] = CoumarinType_Encoder.fit_transform(data['Coumarin Type'])
data['Cancer Type'] = CancerType_Encoder.fit_transform(data['Cancer Type'])

# Filter out the unneeded times for the general data
data = data[
    ((data['Coumarin Type'] == 'Auraptene') & data['Time'].isin(allowed_times_ForAuraptene)) |
    ((data['Coumarin Type'] != 'Auraptene') & data['Time'].isin(allowed_times_NoAuraptene))
]

# Set your X and y here, incase you want general training or training on a specific coumarin.

X = data[['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time']].copy(); # Order must be ['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time']
y = data['Viability'].copy(); # Viability column

# -------------------------
# Load your models
# -------------------------
hist = load("TrainingResults/Models/HistGradientBoostingRegressor/ModelFile/HistGradientBoostingRegressor.joblib")
gbr = load("TrainingResults/Models/GradientBoostingRegressor/ModelFile/GradientBoostingRegressor.joblib")
xgbr = load("TrainingResults/Models/XGBRegressor/ModelFile/XGBRegressor.joblib")
rf = load("TrainingResults/Models/RandomForestRegressor/ModelFile/RandomForestRegressor.joblib")
svr = load("TrainingResults/Models/SVR/ModelFile/SVR.joblib")
elastic = load("TrainingResults/Models/ElasticNet/ModelFile/ElasticNet.joblib")
ridge = load("TrainingResults/Models/Ridge/ModelFile/Ridge.joblib")
lasso = load("TrainingResults/Models/Lasso/ModelFile/Lasso.joblib")

models = [hist, gbr, xgbr, rf, svr, elastic, ridge, lasso]
model_names = [
    "HistGradientBoostingRegressor", 
    "GradientBoostingRegressor", 
    "XGBRegressor", 
    "RandomForestRegressor",
    "SVR",
    "ElasticNet",
    "Ridge",
    "Lasso"
]

# ---------------------------------------------------
# Your feature names
# ---------------------------------------------------
feature_names = np.array(["Cancer Type", "Coumarin Type", "Coumarin Dose", "Time"])

# ---------------------------------------------------
# Create subplots
# ---------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

# ---------------------------------------------------
# Loop through each model and compute permutation importance
# ---------------------------------------------------
for ax, model, name in zip(axes, models, model_names):

    r = permutation_importance(
        model, X, y,
        n_repeats=20,
        random_state=42
    )

    importances = r.importances_mean
    indices = np.argsort(importances)

    ax.bar(range(len(importances)), importances[indices], color = "#585858")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(feature_names[indices], fontsize=7)
    ax.set_title(f"{name}", fontsize=9)

plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.tight_layout()
plt.show()