import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, make_scorer
import os
import sys
import matplotlib.pyplot as plt
import optuna
from joblib import dump

# Please Change the paths to match your own enviroment.
DATA_PATH = 'Total_Data.csv'

# Change model name here. MAKE THE NAME SAME AS THE CONSTRUCTOR(e.g. GradientBoostingRegressor, RandomForestRegressor, ...)
# Also in case you're doing training on different coumarins, the name the model as the following example: "RandomForestRegressor(AURAPTENE)"
CURRENT_MODEL = "HistGradientBoosting(GENERAL)";

# Wether we intend to do grid searching on the model or not is denoted the variable below:
DO_GRIDSEARCH = False;

# RESULT_DIR = f'/TrainingResults/Models/{CURRENT_MODEL}/'
RESULT_DIR = os.path.join(os.getcwd(), f'TrainingResults/Models/{CURRENT_MODEL}/')
os.makedirs(RESULT_DIR, exist_ok=True)

mainData = pd.read_csv(DATA_PATH);

mainData = mainData[['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time', 'Viability']].dropna();

allowed_times_NoAuraptene = [24, 48, 72]
allowed_times_ForAuraptene = [24, 48, 72, 96]

coumarins = ['Auraptene', 'Esculetin', 'Galbanic Acid', 'Umbelliprenin'];

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
reliable_data = mainData[mainData['Cancer Type'].isin(general_reliable_cancers)].copy()

data = reliable_data;
general_CancerType_Encoder = LabelEncoder();
general_CoumarinType_Encoder = LabelEncoder();

data['Coumarin Type'] = general_CoumarinType_Encoder.fit_transform(data['Coumarin Type'])
data['Cancer Type'] = general_CancerType_Encoder.fit_transform(data['Cancer Type'])

# Filter out the unneeded times for the general data
data = data[
    ((data['Coumarin Type'] == 'Auraptene') & data['Time'].isin(allowed_times_ForAuraptene)) |
    ((data['Coumarin Type'] != 'Auraptene') & data['Time'].isin(allowed_times_NoAuraptene))
]



# Set your X and y here, incase you want general training or training on a specific coumarin.

X = data[['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time']].copy()

y = data['Viability'].copy(); # Viability column

# ====<BEGIN GRID SEARCHING>====
# Grid searching is done here, the program stops after grid searching and the parameters and the scores are all saved in reports directory:
if DO_GRIDSEARCH :
    
    gs_Estimator = HistGradientBoostingRegressor(random_state=42);
    
    # The parameters for the search.
    gs_parameters = {
        "learning_rate": [0.01,0.03,0.05,0.07],  # 4
        "max_depth": [3,5,7],                    # 3
        "max_leaf_nodes": [31,63,127],           # 3
        "min_samples_leaf": [5,10,20],           # 3
        "l2_regularization": [0.0,0.1,0.5],      # 3
        "loss": ["squared_error", "absolute_error"]
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
params = sanitize_params(HistGradientBoostingRegressor, params=params)

# params['max_depth'] = int(params['max_depth']);
# params['n_estimators'] = int(params['n_estimators']);
params['random_state'] = 42

# feed the params to model, with random_state as 42
model = HistGradientBoostingRegressor(**params);

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


os.makedirs(f"{RESULT_DIR}/ModelFile/", exist_ok=True);
dump(model, f"{RESULT_DIR}/ModelFile/{CURRENT_MODEL}.joblib")

print("==================================<OPTIMIZING>==================================");


coumarins = [
    "Esculetin",
    "Auraptene",
    "Umbelliprenin",
    "Galbanic Acid"
]

colon = general_CancerType_Encoder.transform(["Colon"])[0]

time_points = [24, 48, 72]
results = []

# Optional: plot dose-response curves
# def plot_dose_response(coumarin, time_point, dose_min=0, dose_max=200):
#     doses = np.linspace(dose_min, dose_max, 400)
#     X = pd.DataFrame([{
#         "Cancer Type": colon,
#         "Coumarin Type": general_CoumarinType_Encoder.transform([coumarin])[0],
#         "Coumarin Dose": d,
#         "Time": time_point
#     } for d in doses])
#     viability = model.predict(X)
#     plt.figure()
#     plt.scatter(doses, viability)
#     plt.axhline(50, linestyle="--")
#     plt.xlabel("Dose")
#     plt.ylabel("Predicted Viability")
#     plt.title(f"{coumarin} — {time_point}h")
#     plt.show()


def pick_two_best_windows(coumarin, time_point, target=50, tol=2, max_len=10):
    """
    Returns up to 2 best dose windows based on average deviation from target (50%),
    with each window limited to max_len.
    """
    doses = np.linspace(0, 400, 401)
    X = pd.DataFrame([{
        "Cancer Type": colon,
        "Coumarin Type": general_CoumarinType_Encoder.transform([coumarin])[0],
        "Coumarin Dose": d,
        "Time": time_point
    } for d in doses])
    
    preds = model.predict(X)
    mask = np.abs(preds - target) <= tol

    # --- find contiguous ranges ---
    ranges = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            ranges.append((start, i-1))
            start = None
    if start is not None:
        ranges.append((start, len(doses)-1))
    
    if not ranges:
        return None

    # --- split ranges longer than max_len ---
    candidate_subranges = []
    for s, e in ranges:
        start_idx = s
        while start_idx <= e:
            end_idx = min(start_idx + max_len, e)
            candidate_subranges.append((start_idx, end_idx))
            start_idx = end_idx + 1

    # --- score sub-ranges ---
    scored_subranges = []
    for s, e in candidate_subranges:
        v_slice = preds[s:e+1]
        score = np.sum(np.abs(v_slice - target)) / (doses[e] - doses[s] + 1e-6)
        scored_subranges.append({
            "range": (doses[s], doses[e]),
            "score": score
        })

    # --- pick up to 2 best sub-ranges ---
    scored_subranges.sort(key=lambda x: x["score"])
    best_two = [x["range"] for x in scored_subranges[:2]]
    return best_two


def viability_range(coumarin, time_point):
    doses = np.linspace(0, 400, 401)
    X = pd.DataFrame([{
        "Cancer Type": colon,
        "Coumarin Type": general_CoumarinType_Encoder.transform([coumarin])[0],
        "Coumarin Dose": float(d),
        "Time": time_point
    } for d in doses])
    preds = model.predict(X)
    return preds.min(), preds.max()


# Optional: quick check of viability ranges
for coumarin in coumarins:
    for t in time_points:
        vmin, vmax = viability_range(coumarin, t)
        print(f"{coumarin}, {t}h → [{vmin:.1f}, {vmax:.1f}]")



def format_ranges(ranges):
    if ranges is None:
        return "Not achievable"
    return "; ".join([f"{a:.1f}-{b:.1f}" for a, b in ranges])

final_results = []

def format_ranges(ranges):
    if ranges is None:
        return "Not achievable"
    return "; ".join([f"{a:.1f}-{b:.1f}" for a, b in ranges])

for coumarin in coumarins:
    for t in time_points:
        dranges = pick_two_best_windows(coumarin, t)
        formatted = format_ranges(dranges)

        final_results.append({
            "Cancer Type": "Colon",
            "Coumarin": coumarin,
            "Time (h)": t,
            "Dose Ranges for Viability 50±2": formatted
        })

df_final = pd.DataFrame(final_results)
print(df_final)
os.makedirs(f"{RESULT_DIR}OptimizationWindows", exist_ok=True)
df_final.to_csv(f"{RESULT_DIR}OptimizationWindows/{CURRENT_MODEL}_BestTwoWindows.csv", index=False)