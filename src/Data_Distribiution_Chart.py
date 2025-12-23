import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt;
import matplotlib.cm as cm;
import squarify;

DATA = pd.read_csv("Total_Data.csv");

print(DATA.columns);

# ====<Donut chart for the coumarin distribution in the dataset>====
percentages = DATA["Coumarin Type"].value_counts(normalize=True) * 100;
colors = cm.Blues(np.linspace(0.4, 0.9, len(percentages)))

fig_Coumarin = plt.figure();
ax = fig_Coumarin.add_subplot();
labels = ['Esculetin', 'Umbelliprenin', 'Auraptene', 'Galbanic acid'];
# print(percentages.index)
ax.pie(percentages, labels=labels, 
       autopct='%1.1f%%', startangle=90,
       wedgeprops={'width': 0.5}, pctdistance=0.75,
       colors=colors
       )

ax.set_title("Coumarin distribution");
ax.axis('equal');
plt.show()

# ====<chart for the Cancer Type distribution in the dataset>====

# percentages = DATA["Cancer Type"].value_counts(normalize=True)*100;
# colors = cm.winter(np.linspace(0.4, 0.9, len(percentages)))

# fig_CancerType = plt.figure(figsize=(16,9));
# ax = fig_CancerType.add_subplot();

# bars = ax.bar(percentages.index, percentages.values, color = colors);

# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}%', ha='center')

# ax.set_ylabel('Percentage (%)')
# ax.set_title('Cancer Type Distribution')
# plt.xticks(rotation=45)  # rotate labels if long
# plt.tight_layout()

percentages = DATA["Cancer Type"].value_counts(normalize=True) * 100

# Colors
colors = cm.plasma(np.linspace(0.4, 0.9, len(percentages)))

# Plot
plt.figure(figsize=(16, 9))
xaxis = ['Colorectal', 'Leukemia\nLymphoma', 'Prostate', 'Lung', 'Breast', 'Gastric',
       'Pancreatic', 'Skin', 'Oral', 'Glioma', 'Liver', 'Ovarian', 'Renal',
       'Bone', 'Biliary\nTract', 'Cervical', 'Salivary\nGland', 'Melanoma']
bars = plt.bar(
    xaxis,
    percentages.values,
    color=colors,
)

# Add percentage labels on top of bars
for bar, value in zip(bars, percentages.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}%",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title("Cancer Type Distribution", fontsize=16)
plt.xlabel("Cancer Type", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
# plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()