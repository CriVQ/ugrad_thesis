import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import re

df = pd.read_excel("kontson_shoulderdata_raw.xlsx", sheet_name=1, header=0) # adduction
"""df = pd.read_excel("kontson_shoulderdata_raw.xlsx", sheet_name=0, header=1)""" #flexion
df.columns = df.columns.map(str).str.strip()
participant_cols = [c for c in df.columns if re.fullmatch(r"S([1-9]|1[0-9])", c)]

percent = np.linspace(0, 100, 101)
normalized = []
for col in participant_cols:
    y = df[col].dropna().values
    f = interp1d(np.linspace(0,100,len(y)), y, kind='linear')
    normalized.append(f(percent))

arr            = np.vstack(normalized)
mean_adduction = arr.mean(axis=0)
sd_adduction   = arr.std(axis=0)
kontson_loess  = lowess(mean_adduction, percent, frac=0.1, return_sorted=False)


sbbt = pd.read_excel("raw_shFlex_processed.xlsx")

plt.figure(figsize=(14,8))

# Kontson curve
plt.plot(percent, kontson_loess,
         color='blue', linewidth=2,
         label='Kontson Adduction (LOESS)')
plt.fill_between(percent,
                 mean_adduction - sd_adduction,
                 mean_adduction + sd_adduction,
                 color='blue', alpha=0.2,
                 label='Kontson ±1 SD')

#sBBT curve
plt.plot(sbbt['Percent'], sbbt['LOESS_Dev90'],
         color='red', linewidth=2,
         label='sBBT Abduction (LOESS)')

"""plt.plot(sbbt['Percent'], sbbt['Mean_Flexion_LOESS'],"""

y1 = kontson_loess
y2 = sbbt['LOESS_Dev90'].values

# 2) Compute global min/max (with a little padding)
ymin = min(y1.min(), y2.min()) - 1
ymax = max(y1.max(), y2.max()) + 1



plt.xlabel('Percent of Movement Cycle (%)')
plt.ylabel('Shoulder Adduction Angle (°)')
plt.title('Shoulder Adduction: Kontson vs. sBBT')
plt.ylim(ymin, ymax)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
