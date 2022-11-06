import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi
import seaborn

import matplotlib.dates as mdates
from datetime import date
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

statistic = "gnrha_rate"
years = "2011_2020"
country = "nzl"
name = f"{country}_{statistic}_{years}"


gnrha_source = '2021_09_16_OIA_hormone_medicines_pharmac.xlsx'
pop_source = 'DPE403903_20220806_031634_89.csv'
drugs = pd.read_excel(
        f"./data/{gnrha_source}", 
        skiprows=8, 
        skipfooter=50,
        header=0,
        usecols=range(1, 22)
    ) 

drugs[['year', 'drug']] = drugs[['Financial_year', 'Chemical_group']].fillna(method="ffill")
drugs.set_index(['drug', 'year', 'gender'], inplace=True)
drugs.drop(labels=['Financial_year', 'Chemical_group'], axis=1, inplace=True)

mapper = { f"{y}/{y-1999:02}": f"{y}-07-01" for y in range(2006, 2021) }
drugs.rename(index=mapper, inplace=True)
gnrha = drugs.query("drug == 'GnRH_analogues'")
gnrha = gnrha.sort_index(ascending=True) 
gnrha = gnrha.sort_index(axis=1, ascending=True) 
gnrha = gnrha.fillna(0)

# Assume reported patients '< 10' with 'Unknown' sex was 1  
unknown = gnrha.query("gender == 'Unknown'")
unknown = unknown.replace("< 10", 1)
gnrha.update(unknown)

# Assume reported patients '< 10' with 'Gender diverse' sex was 1  
gd = gnrha.query("gender == 'Gender diverse'")
gd = gd.replace("< 10", 1)
gnrha.update(gd)

# Assume all other '< 10' are 1 
gnrha = gnrha.replace("< 10", 5)

pop = pd.read_csv(
        f"./data/{pop_source}",
        skiprows=2, 
        skipfooter=48,
        index_col=0,
        header=None,
        engine="python"
        )

def to_int(years):
    return int(years.replace(" Years", ""))

pop = pop.fillna(method="ffill", axis=1)
sex_age = [pop.iloc[0].tolist(), map(to_int, pop.iloc[1].tolist())]
cols = list(zip(*sex_age))
index_cols = pd.MultiIndex.from_tuples(cols, names=['sex', 'age'])

pop.index = pop.index.map(lambda x: f"{x}-07-01")
pop.columns = index_cols
pop = pop.iloc[2:]
pop.sort_index(axis=1, inplace=True, ascending=True) 
pop.sort_index(inplace=True, ascending=True) 
pop = pop.astype(int)

female_pop_0_17 = pop.loc[:, ("Female", slice(None))]
male_pop_0_17 = pop.loc[:, ("Male", slice(None))]
pop_0_17 = pop.loc[:, ("Total", slice(None))]

female_pop_0_17.columns = female_pop_0_17.columns.droplevel()
male_pop_0_17.columns = male_pop_0_17.columns.droplevel()
pop_0_17.columns = pop_0_17.columns.droplevel()

df = gnrha.sum(axis=1).to_frame().unstack().fillna(value=0).reset_index(level=0, drop=True)
df.columns = df.columns.droplevel()

df["total_gnrha"] = pd.to_numeric(df.sum(axis=1), downcast="integer")
df["pop_f_0_17"] = pd.to_numeric(female_pop_0_17.sum(axis=1), downcast="integer")
df["pop_m_0_17"] = pd.to_numeric(male_pop_0_17.sum(axis=1), downcast="integer")
df["pop_0_17"] = pop_0_17.sum(axis=1)
df["m_rate_per_100k"] = (df["Male"] / df["pop_m_0_17"]) * 100000
df["f_rate_per_100k"] = (df["Female"] / df["pop_f_0_17"]) * 100000
df["rate_per_100k"] = (df["total_gnrha"] / df["pop_0_17"]) * 100000
df["mean_rate_2006_2009"] = df.loc["2006-07-01":"2009-07-01", "rate_per_100k"].mean() 
df['cpp_other_n'] = pd.to_numeric(round((df['mean_rate_2006_2009'] / 100000) * df['pop_0_17']), downcast="integer")
df['total_excess'] = (df['total_gnrha'] - df['cpp_other_n'])
df['total_gnrha_gd'] = df["total_excess"]
df.loc["2006-07-01":"2009-07-01", "total_gnrha_gd"] = 0

def incidence_from_prevalence(series, window):
    incidences = []
    for i, prevalence in enumerate(series): 
        if i > (window - 2):
            incidence = prevalence - sum([incidences[i-j] for j in range(1, window)])
            incidences.append(incidence)
        else:
            incidences.append(prevalence)

    return incidences

df['3yr_duration_incidence'] = incidence_from_prevalence(df['total_gnrha_gd'], 3)
df['4yr_duration_incidence'] = incidence_from_prevalence(df['total_gnrha_gd'], 4)
df['5yr_duration_incidence'] = incidence_from_prevalence(df['total_gnrha_gd'], 5)
df['cumsum_3'] = df['3yr_duration_incidence'].cumsum()
df['cumsum_4'] = df['4yr_duration_incidence'].cumsum()
df['cumsum_5'] = df['5yr_duration_incidence'].cumsum()
df['pop_9_17'] = pop_0_17.loc[:, 9:17].sum(axis=1)
df['pop_period_start'] = df.loc["2010-07-01", "pop_9_17"]
df.loc["2006-07-01":"2009-7-01", 'pop_period_start'] = None

df['cum_inc_per_9_17_100k'] = (df['cumsum'] / df['pop_period_start']) * 100000
df['prev_per_9_17_100k'] = (df['total_gnrha_gd'] / df['pop_9_17']) * 100000


df.to_csv(f"./results/{name}.csv", float_format="%.2f")

print(df)






 



