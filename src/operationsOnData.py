import pandas as pd;
import numpy as np;

df = pd.read_csv("/home/arashp/Programming_Files/ML_Paper/Total_Data.csv");
df = pd.DataFrame(df);

count = len(df[["Seo", "Year"]].drop_duplicates());
print(count)

print(df['Cancer Type'].value_counts())

print("========================================================================")
print("========================================================================")
print("========================================================================")

data = df[['Coumarin Type', 'Time']];
data = data[data['Coumarin Type'] == "Auraptene"];
print(data.value_counts())