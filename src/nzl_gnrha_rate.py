import matplotlib.pyplot as plt
from numpy import cumsum
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

mapper = { f"{y}/{y-1999:02}": f"{y}-01-01" for y in range(2006, 2021) }
drugs.rename(index=mapper, inplace=True)
gnrha = drugs.query("drug == 'GnRH_analogues'")
gnrha = gnrha.sort_index(ascending=True) 
gnrha = gnrha.sort_index(axis=1, ascending=True) 
gnrha = gnrha.fillna(0)

# drop ages 0-11
gnrha = gnrha.drop(columns=[age for age in range(0, 12)])

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


pop.index = pop.index.map(lambda x: f"{x}-01-01")
pop.columns = index_cols
pop = pop.iloc[2:]
pop.sort_index(axis=1, inplace=True, ascending=True) 
pop.sort_index(inplace=True, ascending=True) 
pop = pop.astype(int)

female_pop_0_17 = pop.loc[:, ("Female", slice(None))]
male_pop_0_17 = pop.loc[:, ("Male", slice(None))]
pop_0_17 = pop.loc[:, ("Total", slice(None))]


print(female_pop_0_17.loc[:, ("Female", slice(12, 17))])

female_pop_0_17.columns = female_pop_0_17.columns.droplevel()
male_pop_0_17.columns = male_pop_0_17.columns.droplevel()
pop_0_17.columns = pop_0_17.columns.droplevel()

female_pop_12_17 = female_pop_0_17.loc[:, 12:17]
male_pop_12_17 = male_pop_0_17.loc[:, 12:17]
pop_12_17 = pop_0_17.loc[:, 12:17]


df = gnrha.sum(axis=1).to_frame().unstack().fillna(value=0).reset_index(level=0, drop=True)
df.columns = df.columns.droplevel()


df["total_gnrha"] = pd.to_numeric(df.sum(axis=1), downcast="integer")


df["pop_f_12_17"] = pd.to_numeric(female_pop_12_17.sum(axis=1), downcast="integer")
df["pop_m_12_17"] = pd.to_numeric(male_pop_12_17.sum(axis=1), downcast="integer")
df["pop_12_17"] = pop_12_17.sum(axis=1)
df["m_rate_per_100k"] = (df["Male"] / df["pop_m_12_17"]) * 100000
df["f_rate_per_100k"] = (df["Female"] / df["pop_f_12_17"]) * 100000
df["rate_per_100k"] = (df["total_gnrha"] / df["pop_12_17"]) * 100000
df["mean_rate_2006_2009"] = df.loc["2006-01-01":"2009-01-01", "rate_per_100k"].mean() 
df['other_n'] = pd.to_numeric(round((df['mean_rate_2006_2009'] / 100000) * df['pop_12_17']), downcast="integer")
df['total_gnrha_gd'] = df["total_gnrha"]
df.loc["2006-01-01":"2009-01-01", "total_gnrha_gd"] = 0

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
df['pop_12_17'] = pop_0_17.loc[:, 12:17].sum(axis=1)
df['pop_6_17'] = pop_0_17.loc[:, 6:17].sum(axis=1)
df['pop_period_start_2008'] = df.loc["2008-01-01", "pop_12_17"]
df['pop_period_start_2009'] = df.loc["2009-01-01", "pop_12_17"]
df['pop_period_start_2017'] = df.loc["2017-01-01", "pop_12_17"]
df['pop_period_start_6_17_2017'] = df.loc["2017-01-01", "pop_6_17"]


df['cum_3yr_inc_per_12_17_100k_2008'] = (df['cumsum_3'] / df['pop_period_start_2008']) * 100000
df['cum_3yr_inc_per_12_17_100k_2009'] = (df['cumsum_3'] / df['pop_period_start_2009']) * 100000
df['cum_4yr_inc_per_12_17_100k_2008'] = (df['cumsum_4'] / df['pop_period_start_2008']) * 100000
df['cum_4yr_inc_per_12_17_100k_2009'] = (df['cumsum_4'] / df['pop_period_start_2009']) * 100000


df['cumsum_3_2017_2021'] = df.loc['2016-01-01':'2020-01-01', '3yr_duration_incidence'].cumsum()

df['cum_3yr_inc_per_12_17_100k_2017'] = (df['cumsum_3_2017_2021'] / df['pop_period_start_2017']) * 100000
# df['cum_3yr_inc_per_6_17_100k_2017'] = (df['cumsum_3_2017_2021'] / df['pop_period_start_6_17_2017']) * 100000



df['prev_per_12_17_100k'] = (df['total_gnrha_gd'] / df['pop_12_17']) * 100000


df.to_csv(f"./results/{name}.csv", float_format="%.2f")


# ## Figures and Tables

seaborn.set_theme()

y_years = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2021)])

# df1 = pd.DataFrame({
#     'total_gnrha': df['total_gnrha'].tolist(),
#     'f_12_17': gnrha.loc[("GnRH_analogues", slice(None), "Female"), 0:17].sum(axis=1).tolist(),
#     'm_0_17': gnrha.loc[("GnRH_analogues", slice(None), "Male"), 0:17].sum(axis=1).tolist(),
#                     },
#                     index=y_years)

# df1.index.name = 'year'

ticks = [f"{x}-01-1" for x in range(2006, 2022, 2)]
labels = [f"{x}/{x-1999:02}" for x in range(2006, 2022, 2)]

# print(df1)

df2 = pd.DataFrame({
        'total_gnrha': df['total_gnrha'].tolist(),
        'modeled_other': df['other_n'].tolist(),
        'total_gnrha_gd':  df['total_gnrha_gd'].tolist()
    }, index=y_years)

df2.index.name = "year"


df3 = pd.DataFrame({
    '3yr_duration_incidence': df['3yr_duration_incidence'].tolist(),
    '4yr_duration_incidence': df['4yr_duration_incidence'].tolist(),
    '3yr_cumumative_inc': df['cumsum_3'].tolist(),
    '4yr_cumumative_inc': df['cumsum_4'].tolist(),
    }, index=y_years)

df4 = pd.DataFrame({
    # 'f_0_9': gnrha.loc[("GnRH_analogues", slice(None), "Female"), 0:9].sum(axis=1).tolist(),
    # 'f_10_11': gnrha.loc[("GnRH_analogues", slice(None), "Female"), 10:11].sum(axis=1).tolist(),
    'f_12_17': gnrha.loc[("GnRH_analogues", slice(None), "Female")].sum(axis=1).tolist(),
    # 'm_0_9': gnrha.loc[("GnRH_analogues", slice(None), "Male"), 0:9].sum(axis=1).tolist(),
    # 'm_10_11': gnrha.loc[("GnRH_analogues", slice(None), "Male"), 10:11].sum(axis=1).tolist(),
    'm_12_17': gnrha.loc[("GnRH_analogues", slice(None), "Male")].sum(axis=1).tolist(),

    }, index=y_years)

df4.index.name = "year"

# print(df4)

# df4.to_csv(f"./results/nzl_gnrha_age_sex_alt.csv")

# print(df4)


# seaborn.lineplot(x='year', y='total_gnrha', data=df1)
seaborn.lineplot(data=df2)
# seaborn.lineplot(x="year", y='f_12_17', data=df4, color="#1f78b4", linestyle="solid", label="Females 12-17")

# # Fig 1
# seaborn.lineplot(x="year", y='total_gnrha', data=df1, color="#9467bd", linestyle="solid", label="Total 12-17")
# seaborn.lineplot(x="year", y='f_12_17', data=df1, color="#1f78b4", linestyle="dashed", label="Females 12-17")
# seaborn.lineplot(x="year", y='m_12_17', data=df1, color="#e31a1c", linestyle="dotted", label="Males 12-17")

plt.ylim([0, 500])
plt.yticks(ticks=[100, 200, 300, 400, 500])


# # Fig 2
# seaborn.lineplot(x="year", y='f_12_17', data=df4, color="#1f78b4", linestyle="dashed", label="Females 12-17")
# seaborn.lineplot(x="year", y='f_10_11', data=df4, color="#1f78b4", linestyle="solid", label="Females 10-11")
# seaborn.lineplot(x="year", y='f_0_9', data=df4, color="#1f78b4", linestyle="dotted", label="Females 0-9")
# seaborn.lineplot(x="year", y='m_12_17', data=df4, color="#e31a1c", linestyle="dashed", label="Males 12-17")
# seaborn.lineplot(x="year", y='m_10_11', data=df4, color="#e31a1c", linestyle="solid", label="Males 10-11")
# seaborn.lineplot(x="year", y='m_0_9', data=df4, color="#e31a1c", linestyle="dotted", label="Males 0-9")
# plt.yticks(ticks=[50, 100, 150, 200, 250])
# plt.ylim([0, 250])

plt.xticks(ticks=ticks, labels=labels)
plt.rc('font', size=12)  
plt.xlabel(None)
plt.ylabel(None)
plt.xlim(["2006-01-01", "2020-01-01"])
plt.legend(loc='upper left')





# df1.style
# dfi.export(df1, 'df_styled.png')


plt.show()







#  



