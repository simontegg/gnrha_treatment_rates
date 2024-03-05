from ipywidgets import Label
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

statistic = "gnrha_rate"
years = "2009_2023"
country = "nzl"
name = f"latest_{country}_{statistic}_{years}"

pop_source = 'DPE403903_20240213_075850_0.csv'
gnrha_source = "2024-02-GnRH-Agonist-statistics.xlsx"

y_years = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2024)])

gnrha = pd.read_excel(
        f"./data/{gnrha_source}", 
        skiprows=15, 
        header=0,
    ) 

df = gnrha.drop(gnrha.columns[0], axis=1)

columns_12_17 = ["Name", "Dispensed_Year", "Age_Band", "Gender", "Patients"]
columns_0_17 = ["Name", "Dispensed_Year", "Gender",	"Patients"]

prev_0_17 = df.iloc[:, :4]  # Columns before the 5th column
prev_0_17.metadata_name = "prev_0_17"
prev_12_17 = df.iloc[:, 5:10]
prev_12_17.metadata_name = "prev_12_17"
prev_12_17.columns = columns_12_17
inc_0_17 = df.iloc[:, 11:15]
inc_0_17.columns = columns_0_17
inc_0_17.metadata_name = "inc_0_17"
inc_12_17 = df.iloc[:, 16:]
inc_12_17.columns = columns_12_17
inc_12_17.metadata_name = "inc_12_17"

male_q = "Gender == 'Male'"
female_q = "Gender == 'Female'"

raw_dfs = [
        prev_0_17,
        prev_12_17,
        inc_0_17,
        inc_12_17
        ]

dfs = {}
for df in raw_dfs:
    m = df.query(male_q).reset_index()
    m = m.replace("≤6", 3)

    if len(m) < len(y_years):
        first_row = m.iloc[0:1]
        new_rows = pd.concat([first_row, first_row], ignore_index=True)
        m = pd.concat([new_rows, m], ignore_index=True)
        m.loc[0:1, 'Patients'] = 0

    m.index = y_years
    dfs[f"{df.metadata_name}_male"] = m

    f = df.query(female_q).reset_index()
    f = f.replace("≤6", 3)
    f.index = y_years
    dfs[f"{df.metadata_name}_female"] = f


inc = pd.DataFrame({
        'year': y_years,
        "males_12_17": dfs["inc_12_17_male"]["Patients"],
        "females_12_17": dfs["inc_12_17_female"]["Patients"],
        "males_0_11": dfs["inc_0_17_male"]["Patients"] - dfs["inc_12_17_male"]["Patients"],
        "females_0_11": dfs["inc_0_17_female"]["Patients"] - dfs["inc_12_17_female"]["Patients"],
    }, index=y_years)

prev = pd.DataFrame({
        'year': y_years,
        "males_12_17": dfs["prev_12_17_male"]["Patients"],
        "females_12_17": dfs["prev_12_17_female"]["Patients"],
        "males_0_11": dfs["prev_0_17_male"]["Patients"] - dfs["prev_12_17_male"]["Patients"],
        "females_0_11": dfs["prev_0_17_female"]["Patients"] - dfs["prev_12_17_female"]["Patients"],
    }, index=y_years)

prev["total_12_17"] = prev["males_12_17"] + prev["females_12_17"]
prev["total_0_11"] = prev["males_0_11"] + prev["females_0_11"]

carryover = pd.DataFrame({
        'year': y_years,
        "males": prev["males_12_17"] - inc["males_12_17"],
        "females": prev["females_12_17"] - inc["females_12_17"],
    })


m_durations = pd.DataFrame({
        'year': y_years,
        "males_12_17_prev": prev["males_12_17"],
        "males_12_17_prev_2": inc["males_12_17"].rolling(window=2, min_periods=1).sum(),
        "males_12_17_prev_3": inc["males_12_17"].rolling(window=3, min_periods=1).sum(),
        "males_12_17_prev_4": inc["males_12_17"].rolling(window=4, min_periods=1).sum(),
        "males_12_17_prev_5": inc["males_12_17"].rolling(window=5, min_periods=1).sum(),
    })

m_durations_0_11 = pd.DataFrame({
        'year': y_years,
        "males_0_11_prev": prev["males_0_11"],
        "males_0_11_prev_2": inc["males_0_11"].rolling(window=2, min_periods=1).sum(),
        "males_0_11_prev_3": inc["males_0_11"].rolling(window=3, min_periods=1).sum(),
        "males_0_11_prev_4": inc["males_0_11"].rolling(window=4, min_periods=1).sum(),
        "males_0_11_prev_5": inc["males_0_11"].rolling(window=5, min_periods=1).sum(),
    })

f_durations = pd.DataFrame({
        'year': y_years,
        "females_12_17_prev": prev["females_12_17"],
        "females_12_17_prev_2": inc["females_12_17"].rolling(window=2, min_periods=1).sum(),
        "females_12_17_prev_3": inc["females_12_17"].rolling(window=3, min_periods=1).sum(),
        "females_12_17_prev_4": inc["females_12_17"].rolling(window=4, min_periods=1).sum(),
        "females_12_17_prev_5": inc["females_12_17"].rolling(window=5, min_periods=1).sum(),
    })

f_durations_0_11 = pd.DataFrame({
        'year': y_years,
        "females_0_11_prev": prev["females_0_11"],
        "females_0_11_prev_2": inc["females_0_11"].rolling(window=2, min_periods=1).sum(),
        "females_0_11_prev_3": inc["females_0_11"].rolling(window=3, min_periods=1).sum(),
        "females_0_11_prev_4": inc["females_0_11"].rolling(window=4, min_periods=1).sum(),
        "females_0_11_prev_5": inc["females_0_11"].rolling(window=5, min_periods=1).sum(),
    })


print(inc)
# m_durations_0_11.plot(kind="line", x="year")

seaborn.set_theme()

total_c = seaborn.color_palette("dark")[7]
f_c = seaborn.color_palette()[3]
m_c = seaborn.color_palette()[0]


# # # Fig 1 Incidence
# seaborn.lineplot(x="year", y='females_12_17', data=inc, color=f_c, linestyle="solid", label="F 12-17")
# seaborn.lineplot(x="year", y='females_0_11', data=inc, color=f_c, linestyle="solid", label="F 0-11")
# seaborn.lineplot(x="year", y='males_12_17', data=inc, color=m_c, linestyle="solid", label="M 12-17")
# seaborn.lineplot(x="year", y='males_0_11', data=inc, color=m_c, linestyle="solid", label="M 0-11")
# plt.ylim([0, 120])
# plt.yticks(ticks=[20, 40, 60, 80, 100, 120])
# plt.ylabel("Treatment incidence")

# Prevalence
# seaborn.lineplot(x="year", y='total_12_17', data=prev, color=total_c, linestyle="solid", label="Total 12-17")
# seaborn.lineplot(x="year", y='total_0_11', data=prev, color=total_c, linestyle="dashed", label="Total 0-11")
# seaborn.lineplot(x="year", y='females_12_17', data=prev, color=f_c, linestyle="solid", label="F 12-17")
# seaborn.lineplot(x="year", y='females_0_11', data=prev, color=f_c, linestyle="dashed", label="F 0-11")
# seaborn.lineplot(x="year", y='males_12_17', data=prev, color=m_c, linestyle="solid", label="M 12-17")
# seaborn.lineplot(x="year", y='males_0_11', data=prev, color=m_c, linestyle="dashed", label="M 0-11")
# plt.ylim([0, 450])
# plt.yticks(ticks=[50, 100, 150, 200, 250, 300, 350, 400, 450])
# plt.ylabel("Treatment prevalence")


# General
# ticks = [f"{x}-01-1" for x in range(2006, 2025, 2)]
# labels = [f"{x}" for x in range(2006, 2025, 2)]
# plt.xticks(ticks=ticks, labels=labels)
# plt.rc('font', size=12)  
# plt.xlabel(None)
# plt.xlim(["2006-01-01", "2024-01-01"])
# plt.legend(loc='upper left')
# plt.show()


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

female_pop_0_17.columns = female_pop_0_17.columns.droplevel()
male_pop_0_17.columns = male_pop_0_17.columns.droplevel()
pop_0_17.columns = pop_0_17.columns.droplevel()

female_pop_12_17 = female_pop_0_17.loc[:, 12:17]
male_pop_12_17 = male_pop_0_17.loc[:, 12:17]
pop_12_17 = pop_0_17.loc[:, 12:17]
pop_6_17 = pop_0_17.loc[:, 6:17]
pop_12_17.index = y_years
pop_6_17.index = y_years

df = pd.DataFrame({
    'year': y_years,
    'males_0_11': inc["males_0_11"],
    'females_0_11': inc["females_0_11"],
    'total_0_11': inc["males_0_11"] + inc["females_0_11"],
    'males_12_17': inc["males_12_17"],
    'females_12_17': inc["females_12_17"],
    'total_12_17': inc['males_12_17'] + inc['females_12_17'],
    })

df["gd_cumulative_incidence"] = df["total_12_17"]["2010-01-01":].cumsum()


df["pop_12_17"] = pop_12_17.sum(axis=1)
df['pop_6_17'] = pop_6_17.sum(axis=1)
df['pop_period_start_2008'] = df.loc["2008-01-01", "pop_12_17"]
df['pop_period_start_2009'] = df.loc["2009-01-01", "pop_12_17"]
df['pop_period_start_2016'] = df.loc["2016-01-01", "pop_12_17"]
df['pop_period_start_2017'] = df.loc["2017-01-01", "pop_12_17"]
df['pop_period_start_6_17_2017'] = df.loc["2017-01-01", "pop_6_17"]


df['cum_inc_per_12_17_100k_2008'] = (df['gd_cumulative_incidence'] / df['pop_period_start_2008']) * 100000
df['cum_inc_per_12_17_100k_2009'] = (df['gd_cumulative_incidence'] / df['pop_period_start_2009']) * 100000
df['cum_inc_2017_2021'] = df.loc['2017-01-01':'2021-01-01', 'gd_cumulative_incidence'].cumsum()
df['cum_inc_per_12_17_100k_2017'] = (df['cum_inc_2017_2021'] / df['pop_period_start_2017']) * 100000


df.to_csv(f"./results/{name}.csv", float_format="%.2f")

print(df)

# print(df)




# # ## Figures and Tables
# seaborn.set_theme()

# # y_years = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2023)])

# # # https://www.engage.england.nhs.uk/consultation/puberty-suppressing-hormones/user_uploads/engagement-report-interim-policy-on-puberty-suppressing-hormones-for-gender-incongruence-or-dysphoria.pdf
# # eng_wls = ([np.nan] * (2022 - 2006)) + [378] 
# # hue = (["first"] * (2023 - 2006))


# # ticks = [f"{x}-01-1" for x in range(2006, 2024, 2)]
# # labels = [f"{x}" for x in range(2006, 2024, 2)]

# # eng_wls_rate = ([np.nan] * (2022 - 2006)) + [9.6] 

# # df2 = pd.DataFrame({
# #     # 'nz_12_17_rate': (df1['nz_12_17'] / pop_12_17_.sum(axis=1)) * 100000,
# #     'eng_wls_12_17_rate': pd.Series(data=eng_wls_rate, index=y_years),
# #     }, index=y_years)

# # 'gnrha_prevalence_12_17'
#     # 'gnrha_prevalence_0_17': total_0_17



# colors = seaborn.color_palette("bright")
# # colors.reverse()


# # # # Fig 1
# # seaborn.lineplot(x="year", y='gnrha_prevalence_0_17', data=df, color=colors[0], linestyle="solid", label="Minors 0-17")
# # seaborn.lineplot(x="year", y='gnrha_prevalence_12_17', data=df, color=colors[3], linestyle="solid", label="Adolescents 12-17")
# # seaborn.lineplot(x="year", y='total_0_11', data=df, color=colors[2], linestyle="solid", label="Children 0-11")
# # plt.ylim([0, 700])
# # plt.yticks(ticks=[100, 200, 300, 400, 500, 600, 700])


# # ## Rate
# # # seaborn.lineplot(x="years", y='nz_12_17_rate', data=df2, color="#d62728", linestyle="solid", label="NZ adolescents 12-17 ")
# # # seaborn.scatterplot(x="years", y='eng_wls_12_17_rate', data=df2, label="England & Wales adolescents 12-17")



# # # # Fig 2
# seaborn.lineplot(x="year", y='females_12_17', data=df, color="#1f78b4", linestyle="solid", label="Females 12-17")
# seaborn.lineplot(x="year", y='females_0_11', data=df, color="#1f78b4", linestyle="dashed", label="Females 0-11")
# seaborn.lineplot(x="year", y='males_12_17', data=df, color="#e31a1c", linestyle="solid", label="Males 12-17")
# seaborn.lineplot(x="year", y='males_0_11', data=df, color="#e31a1c", linestyle="dashed", label="Males 0-11")
# plt.yticks(ticks=[50, 100, 150, 200, 250])
# plt.ylim([0, 250])


# ticks = [f"{x}-01-1" for x in range(2006, 2024, 2)]
# labels = [f"{x}" for x in range(2006, 2024, 2)]

# plt.xticks(ticks=ticks, labels=labels)
# plt.rc('font', size=12)  
# plt.ylabel("Treatment prevalence")
# plt.xlabel(None)
# plt.xlim(["2006-01-01", "2023-01-01"])
# plt.legend(loc='upper left')






# # # df1.style
# # # dfi.export(df1, 'df_styled.png')


# plt.show()
