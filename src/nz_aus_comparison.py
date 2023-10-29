import matplotlib.pyplot as plt
from numpy import cumsum
import numpy as np
import pandas as pd
import dataframe_image as dfi
import seaborn

import matplotlib.dates as mdates
from datetime import date
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


y_years = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2023)])

nz = [
   5.505164,
    4.47392,
   3.986287,
    6.46047,
   8.903758,
  13.025427,
  16.798071,
  22.796971,
  24.599574,
  30.385984,
  34.338039,
  53.286448,
  59.754522,
  71.802782,
  82.393305,
 101.627073,
 108.124968]

sa = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.82,
        7.45,
        9.11,
        8.21,
        13.83,
        9.67,
        np.nan,
        np.nan,
        np.nan,
        ]

wa = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        1.62,
        0.54,
        7.57,
        14.9,
        18.29,
        25.65,
        24.1,
        21.2,
        22.75
        ]

vc = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        3.31,
        6.09,
        6.68,
        7.47,
        6.67,
        3.71,
        3.24,
        9.14,
        12.58
        ]

ql = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.55,   
        4.39,
        8.15,
        20.23,
        44.63,
        29.41,
        21.88,
        17.42,
        19.52,
        ]

df1 = pd.DataFrame({
    'year': pd.Series(data=y_years, index=y_years),
    "nz_12_17": pd.Series(data=nz, index=y_years),
    "ql": pd.Series(data=ql, index=y_years),
    "wa": pd.Series(data=wa, index=y_years),
    "sa": pd.Series(data=sa, index=y_years),
    "vc": pd.Series(data=vc, index=y_years),
                    },
                    index=y_years)



ticks = [f"{x}-01-1" for x in range(2006, 2024, 2)]
labels = [f"{x}" for x in range(2006, 2024, 2)]


# # Fig 1
seaborn.lineplot(x="year", y='nz_12_17', data=df1, color="#1f77b4", linestyle="solid", label="NZ adolescents 12-17")
seaborn.lineplot(x="year", y='sa', data=df1, color="#ff7f0e", linestyle="--", label="South Australia")
seaborn.lineplot(x="year", y='wa', data=df1, color="#2ca02c", linestyle="-.", label="Western Australia")
seaborn.lineplot(x="year", y='vc', data=df1, color="#d62728", linestyle=":", label="Victoria")
seaborn.lineplot(x="year", y='ql', data=df1, color="#9467bd", linestyle="solid", label="Queensland")

plt.xticks(ticks=ticks, labels=labels)
plt.rc('font', size=12)  
plt.ylabel("Treatment prevalence per 100k")
plt.xlabel(None)
plt.xlim(["2006-01-01", "2023-01-01"])
plt.legend(loc='upper left')

plt.show()

