# THINGS TO MODIFY ARE :
# 1. nameOfModel
# 2. CURRENT_COUMARIN (in case we are doing training on separate coumarins, leave it as None if doing General training)
# 3. SEPARATE_COUMARINS (True if we are doing training on separate coumarins)
# 4. DO_GRIDSEARCH (True if we want to gridSearch, False we are doing training and predicting)
# 5. X and y
# 6. gs_Estimator (the estimator of the gridSearching)
# 7. gs_parameters
# 8. model (Must be given **params as argument
# THIS SHOULD COVER DIFFERENT CASES AND DIFFERENT MODEL TRAININGS
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
import os
import sys
import matplotlib.pyplot as plt
import itertools

# Please Change the paths to match your own enviroment.
DATA_PATH = 'Total_Data.csv'

# Change model name here. MAKE THE NAME SAME AS THE CONSTRUCTOR(e.g. GradientBoostingRegressor, RandomForestRegressor, ...)
# Also in case you're doing training on different coumarins, the name the model as the following example: "RandomForestRegressor(AURAPTENE)"
CURRENT_MODEL = "XGBOOST(GENERAL)";
CURRENT_COUMARIN = None
SEPARATE_COUMARINS = False

# RESULT_DIR = f'/TrainingResults/Models/{CURRENT_MODEL}/'
RESULT_DIR = os.path.join(os.getcwd(), f'TrainingResults/Models/{CURRENT_MODEL}/')
os.makedirs(RESULT_DIR, exist_ok=True)

mainData = pd.read_csv(DATA_PATH);

mainData = mainData[['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time', 'Viability']].dropna();

allowed_times_NoAuraptene = [24, 48, 72]
allowed_times_ForAuraptene = [24, 48, 72, 96]

coumarins = ['Auraptene', 'Esculetin', 'Galbanic Acid', 'Umbelliprenin'];
separateCoumarinDataDict = {"Auraptene" : None, "Esculetin" : None, 
                            "Glabanic Acid" : None, "Umbelliprenin" : None};

# ===========================================================================================================
# Filtering the data using GMM for all Coumarins and the overall data itself and saving them in the dict
# We filter out the unreliable Cancer Types
# ===========================================================================================================


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

general_reliable_cancers = count_df[count_df['Reliability'] == 'Reliable']['Cancer Type'].tolist()
general_reliable_data = mainData[mainData['Cancer Type'].isin(general_reliable_cancers)].copy()

general_Data = general_reliable_data;
general_CancerType_Encoder = LabelEncoder();
general_CoumarinType_Encoder = LabelEncoder();

general_Data['Coumarin Type'] = general_CoumarinType_Encoder.fit_transform(general_Data['Coumarin Type'])
general_Data['Cancer Type'] = general_CancerType_Encoder.fit_transform(general_Data['Cancer Type'])

# Filter out the unneeded times for the general data
general_Data = general_Data[
    ((general_Data['Coumarin Type'] == 'Auraptene') & general_Data['Time'].isin(allowed_times_ForAuraptene)) |
    ((general_Data['Coumarin Type'] != 'Auraptene') & general_Data['Time'].isin(allowed_times_NoAuraptene))
]

# We shall save each encoder of different coumarin type in case we need it in future for decoding
Encoders = {"Auraptene" : None, "Esculetin" : None, 
            "Glabanic Acid" : None, "Umbelliprenin" : None};

# Now filter the data for seperate coumarins:===============================================================================================
for coumarin in coumarins :

    if coumarin != "Auraptene" :
        coumarin_data = mainData[(mainData['Coumarin Type'] == coumarin) & (mainData['Time'].isin(allowed_times_NoAuraptene))].copy()
    else :
        coumarin_data = mainData[(mainData['Coumarin Type'] == coumarin) & (mainData['Time'].isin(allowed_times_ForAuraptene))].copy()

    
    coumarin_data.drop(columns=['Coumarin Type'], inplace=True);

    cancer_counts = coumarin_data['Cancer Type'].value_counts().to_dict()
    count_df = pd.DataFrame(list(cancer_counts.items()), columns=['Cancer Type', 'Sample Count'])

    gmm.fit(count_df[['Sample Count']])
    threshold = np.mean(gmm.means_.flatten())

    count_df['Reliability'] = count_df['Sample Count'].apply(
        lambda x: 'Reliable' if x >= threshold else 'Unreliable'
    )

    reliable_cancers = count_df[count_df['Reliability'] == 'Reliable']['Cancer Type'].tolist()
    reliable_data = coumarin_data[coumarin_data['Cancer Type'].isin(reliable_cancers)].copy()

    # Perfom transformation on the Cancer Type column
    CancerEncoder = LabelEncoder();
    reliable_data['Cancer Type'] = CancerEncoder.fit_transform(reliable_data['Cancer Type']);
    
    # Save the encoder in case we are gonna need it
    Encoders[coumarin] = CancerEncoder;

    # save the new coumarin data set
    separateCoumarinDataDict[coumarin] = reliable_data;
# Done looping through coumarins, we now have the data we need for each case================================================================

# Wether we intend to do grid searching on the model or not is denoted the variable below:
DO_GRIDSEARCH = False;

# Set your X and y here, incase you want general training or training on a specific coumarin.

X = general_Data[['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time']].copy(); # Order must be ['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time'](Coumarin type if only doing general training)
y = general_Data['Viability'].copy(); # Viability column

# ====<BEGIN GRID SEARCHING>====
# Grid searching is done here, the program stops after grid searching and the parameters and the scores are all saved in reports directory:
if DO_GRIDSEARCH :
    
    gs_Estimator = XGBRegressor(random_state=42);
    
    # The parameters for the search.
    gs_parameters = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 1.5]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42);
    gs = GridSearchCV(estimator=gs_Estimator,
                                scoring='r2',
                                param_grid=gs_parameters,
                                cv=kf,
                                n_jobs=-1,
                                verbose=1
                                );
    
    gs.fit(X, y);

    BEST_MODEL_PARAMETERS = gs.best_params_;
    BEST_MODEL_R2_SCORE = gs.best_score_;

    dict = BEST_MODEL_PARAMETERS.copy()
    dict['R2'] = BEST_MODEL_R2_SCORE;
    
    df = pd.DataFrame(dict, index=[0]);

    # Save the best found parameters
    os.makedirs(f"{RESULT_DIR}Best_Paremeters/", exist_ok=True);
    df.to_csv(f"{RESULT_DIR}Best_Paremeters/{CURRENT_MODEL}_GS_BestParameters.csv",index=False);

    sys.exit(0);
# ====<END OF GRID SEARCHING>====

# ====<START EVALUATING MODEL>====
# Get the model parameters post GridSearching
params = pd.read_csv(f"{RESULT_DIR}Best_Paremeters/{CURRENT_MODEL}_GS_BestParameters.csv").iloc[0].to_dict();

#Thanks chatgpt
def sanitize_params(estimator, params):
    """
    Cleans and fixes parameter types before passing them to sklearn models.
    - Converts floats representing ints into int
    - Converts 'None' strings to None
    - Converts NaN to None
    - Removes unexpected params
    """
    import inspect

    valid_params = estimator().get_params().keys()
    clean = {}

    for key, value in params.items():

        # Skip invalid params
        if key not in valid_params:
            continue

        # Convert string 'None' to Python None
        if isinstance(value, str) and value.strip().lower() == "none":
            clean[key] = None
            continue

        # Convert NaN to None
        if isinstance(value, float) and np.isnan(value):
            clean[key] = None
            continue

        # Convert float-like integers to real ints
        if isinstance(value, float) and value.is_integer():
            clean[key] = int(value)
            continue

        # Leave everything else unchanged
        clean[key] = value

    return clean

params.pop("R2");
params = sanitize_params(XGBRegressor, params=params)

# params['max_depth'] = int(params['max_depth']);
# params['n_estimators'] = int(params['n_estimators']);
params['random_state'] = 42

# feed the params to model, with random_state as 42
model = XGBRegressor(**params);

# Start Evaluating Model performance. Get R2, MAE, RMSE
cv_Scores = {"R2" : None, "MAE" : None, "MSE" : None, "MedAE" : None};

kf = KFold(n_splits=5, shuffle=True, random_state=42);

scorers = {
    "R2": "r2",
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
    "MedAE" : make_scorer(median_absolute_error, greater_is_better=False)
}

for score_type, scorer in scorers.items() :
    score = cross_val_score(estimator=model, scoring=scorer, X=X, y=y, cv=kf, n_jobs=-1, verbose=0).mean()
    if score_type in ["MAE", "MSE", "MedAE"] :
        score = -score;
    
    cv_Scores[score_type] = score;

df = pd.DataFrame(cv_Scores, index=[0])

# save the evaluation results
os.makedirs(f"{RESULT_DIR}Evaluations/", exist_ok=True);
df.to_csv(f"{RESULT_DIR}Evaluations/{CURRENT_MODEL}_Evaluation_Report.csv",index=False);
# ====<DONE EVAlUATING THE MODEL>====

# Train the model on our X and y 
model.fit(X,y)

# # ====<Predict Best Dose and Time for viability 50>====

# doseRange = np.linspace(0, 400, 401)

# pred_Dir = f"{RESULT_DIR}Best_Dose_Time_{CURRENT_MODEL}/";
# os.makedirs(pred_Dir, exist_ok=True);


# # Prepare to store best combinations
# best_conditions = []

# # ====<Generate all Dose ,Time combinations>====
# dose_grid, time_grid = np.meshgrid(doseRange, allowed_times_ForAuraptene, indexing='ij')
# dose_flat = dose_grid.ravel()
# time_flat = time_grid.ravel()

# if SEPARATE_COUMARINS == False :
    
#     # Get unique coumarins and cancer types
#     coumarins = general_Data['Coumarin Type'].unique()
#     cancer_types = general_Data['Cancer Type'].unique()
    
#     # Loop over each Cancer, Coumarin pairs
#     for coumarin in coumarins:
#         for cancer in cancer_types:
#             # Build DataFrame for this pair
#             pred_df = pd.DataFrame({
#                 'Cancer Type': [cancer] * len(dose_flat),
#                 'Coumarin Type': [coumarin] * len(dose_flat),
#                 'Coumarin Dose': dose_flat,
#                 'Time': time_flat
#             })

#             # Apply Auraptene rule: only Auraptene can have time=96
#             if coumarin != "Auraptene":
#                 pred_df = pred_df[pred_df['Time'] != 96]

#             # Predict
#             preds = model.predict(pred_df)

#             # Find best Dose/Time (closest to 50)
#             min_idx = np.abs(preds - 50).argmin()
#             best_row = pred_df.iloc[min_idx]

#             # Decode labels
#             cancer_decoded = general_CancerType_Encoder.inverse_transform([int(best_row['Cancer Type'])])[0]
#             coumarin_decoded = general_CoumarinType_Encoder.inverse_transform([int(best_row['Coumarin Type'])])[0]

#             best_conditions.append({
#                 'Coumarin': coumarin_decoded,
#                 'Cancer Type': cancer_decoded,
#                 'Best Dose': best_row['Coumarin Dose'],
#                 'Best Time': best_row['Time'],
#                 'Predicted Viability': preds[min_idx]
#             })

#     # ====<SAVE RESULTS>====
#     df_best = pd.DataFrame(best_conditions)
#     df_best.to_csv(f"{pred_Dir}General_Best_Dose_Time_GridSearch_ByPair.csv", index=False)


# else :

#     # Get cancer types
#     cancer_types = separateCoumarinDataDict[CURRENT_COUMARIN]['Cancer Type'].unique()
    
#     for coumarin in [CURRENT_COUMARIN]:
#         for cancer in cancer_types:
#             # Build DataFrame for this pair
#             pred_df = pd.DataFrame({
#                 'Cancer Type': [cancer] * len(dose_flat),
#                 'Coumarin Dose': dose_flat,
#                 'Time': time_flat
#             })

#             # Apply Auraptene rule: only Auraptene can have time=96
#             if coumarin != "Auraptene":
#                 pred_df = pred_df[pred_df['Time'] != 96]

#             # Predict
#             preds = model.predict(pred_df)

#             # Find best Dose/Time (closest to 50)
#             min_idx = np.abs(preds - 50).argmin()
#             best_row = pred_df.iloc[min_idx]

#             # Decode labels
#             cancer_decoded = Encoders[CURRENT_COUMARIN].inverse_transform([int(best_row['Cancer Type'])])[0]

#             best_conditions.append({
#                 'Coumarin': coumarin,
#                 'Cancer Type': cancer_decoded,
#                 'Best Dose': best_row['Coumarin Dose'],
#                 'Best Time': best_row['Time'],
#                 'Predicted Viability': preds[min_idx]
#             })

#     # ====<SAVE RESULTS>====
#     df_best = pd.DataFrame(best_conditions)
#     df_best.to_csv(f"{pred_Dir}{coumarin}_Best_Dose_Time_GridSearch_ByPair.csv", index=False)

# print("====<DONE PREDICTING>====")

def predict_viability(model, cancer_type, coumarin_type, dose, time, 
                      cancer_encoder, coumarin_encoder):
    """
    Predict viability for a single input or a list of inputs.
    
    Parameters:
    - model: trained sklearn regressor
    - cancer_type: string or list of strings
    - coumarin_type: string or list of strings
    - dose: float or list of floats
    - time: float or list of floats
    - cancer_encoder: LabelEncoder for cancer types
    - coumarin_encoder: LabelEncoder for coumarin types (general) or specific coumarin

    Returns:
    - predicted viability (float or np.array)
    """
    import numpy as np
    import pandas as pd

    # Ensure inputs are lists
    if not isinstance(cancer_type, list):
        cancer_type = [cancer_type]
        coumarin_type = [coumarin_type]
        dose = [dose]
        time = [time]

    # Encode categorical variables
    cancer_encoded = cancer_encoder.transform(cancer_type)
    coumarin_encoded = coumarin_encoder.transform(coumarin_type)

    # Build DataFrame
    df = pd.DataFrame({
        'Cancer Type': cancer_encoded,
        'Coumarin Type': coumarin_encoded,
        'Coumarin Dose': dose,
        'Time': time
    })

    # Predict
    predictions = model.predict(df)

    # If input was a single value, return single prediction
    if len(predictions) == 1:
        return predictions[0]
    return predictions

predicted_viability = predict_viability(
    model=model,
    cancer_type="Colon",        # example cancer type
    coumarin_type="Auraptene", # example coumarin
    dose=75,                    # dose value
    time=72,                    # time value
    cancer_encoder=general_CancerType_Encoder,
    coumarin_encoder=general_CoumarinType_Encoder
)

print(f"Predicted Viability: {predicted_viability:.2f}")

from joblib import dump
os.makedirs(f"{RESULT_DIR}/ModelFile/", exist_ok=True);
dump(model, f"{RESULT_DIR}/ModelFile/{CURRENT_MODEL}.joblib")