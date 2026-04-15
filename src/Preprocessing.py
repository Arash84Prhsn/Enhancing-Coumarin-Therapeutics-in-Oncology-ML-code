import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
import os
from joblib import dump


# Please Change the paths to match your own enviroment.(in case of failure that is)
DATA_PATH = 'Total_Data.csv'

rawData = pd.read_csv(DATA_PATH);

# ====<Pre-Processing and GMM filtering>====
mainData = rawData[['Cancer Type', 'Coumarin Type', 'Coumarin Dose', 'Time', 'Viability']].dropna();

allowed_times_NoAuraptene = [24, 48, 72]
allowed_times_ForAuraptene = [24, 48, 72, 96]

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

os.makedirs("Encoders", exist_ok=True);

dump(CancerType_Encoder, "Encoders/CancerEncoder.joblib")
dump(CoumarinType_Encoder, "Encoders/CoumarinEncoder.joblib")

# Filter out the unneeded times for the general data
data = data[
    ((data['Coumarin Type'] == 'Auraptene') & data['Time'].isin(allowed_times_ForAuraptene)) |
    ((data['Coumarin Type'] != 'Auraptene') & data['Time'].isin(allowed_times_NoAuraptene))
]

data.to_csv("Processed_data.csv", index=False);