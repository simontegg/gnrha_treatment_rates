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
years = "2011_2022"
country = "nzl"
name = f"latest_{country}_{statistic}_{years}"

pop_source = 'DPE403903_20240213_075850_0.csv'

gnrha_source_0_17 = '2023-08_GnRH_Calendar_year.xslx'
gnrha_source_12_17 =  '2023-08_GnRH_12-17_NZ.xlsx'


y_years = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2023)])

gnrha_0_17 = pd.read_excel(
        f"./data/{gnrha_source_0_17}", 
        skiprows=5, 
        header=0,
    ) 

gnrha_12_17 = pd.read_excel(
        f"./data/{gnrha_source_12_17}", 
        skiprows=5, 
        header=0,
    ) 

males_0_17 = gnrha_0_17.query("gender == 'Male'").reset_index()
females_0_17 = gnrha_0_17.query("gender == 'Female'").reset_index()
males_0_17 = males_0_17.replace("<6", 3)
other_0_17 = gnrha_0_17.query("gender == 'Other' or gender == 'Unknown'")
other_0_17 = other_0_17.replace("<6", 3)
summed_other = other_0_17.groupby('Calendar_Year')['Patients'].sum().reset_index()
summed_other.index = pd.to_datetime([f"{y}-01-01" for y in range(2013, 2023)])
new_index = pd.date_range(start='2006-01-01', end=summed_other.index.max(), freq='AS')
summed_other = summed_other.reindex(new_index, fill_value=0)
males_0_17.index = y_years
females_0_17.index = y_years

males_12_17 = gnrha_12_17.query("gender == 'Male'").reset_index()
females_12_17 = gnrha_12_17.query("gender == 'Female'").reset_index()
males_12_17 = males_12_17.replace("<6", 3)
males_12_17.index = y_years
females_12_17.index = y_years

dfs_0_17 = [females_0_17, males_0_17]
dfs_12_17 = [females_12_17, males_12_17, summed_other]

total_0_17 = sum(df['Patients'] for df in dfs_0_17)
total_12_17 = sum(df['Patients'] for df in dfs_12_17)

# print(females_12_17)
# print(males_12_17)




# males = males.drop(labels=[31, 33])
# females = females.drop(labels=[30, 32])
# males = males.reset_index(drop=True)
# females = females.reset_index(drop=True)


# males = males.reindex(labels=mapper)
# females.rename(index=mapper, inplace=True)


## Change back to 2021 max for comparison
# males.index = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2021)])
# m.index = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2023)])
# females.index = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2021)])
# f.index = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2023)])
# y_years = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2021)])

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

# drop 2023
last_row_index = pop.index[-1]
pop = pop.drop(last_row_index)




female_pop_0_17 = pop.loc[:, ("Female", slice(None))]
male_pop_0_17 = pop.loc[:, ("Male", slice(None))]
pop_0_17 = pop.loc[:, ("Total", slice(None))]

female_pop_0_17.columns = female_pop_0_17.columns.droplevel()
male_pop_0_17.columns = male_pop_0_17.columns.droplevel()
pop_0_17.columns = pop_0_17.columns.droplevel()

female_pop_12_17 = female_pop_0_17.loc[:, 12:17]
male_pop_12_17 = male_pop_0_17.loc[:, 12:17]
pop_12_17 = pop_0_17.loc[:, 12:17]
pop_12_17.index = y_years

# pop_12_17['total'] = pop_12_17.sum(axis=1)

# pop_12_17_ = pd.concat([pop_12_17, pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=pop_12_17.columns)], ignore_index=True)
# pop_12_17_ = pd.concat([pop_12_17_, pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=pop_12_17.columns)], ignore_index=True)

# pop_12_17_.index = y_years
# pop_12_17_.interpolate(axis=0, inplace=True, method='time')

def incidence_from_prevalence(series, window):
    incidences = []
    for i, prevalence in enumerate(series): 
        if i > (window - 2):
            incidence = prevalence - sum([incidences[i-j] for j in range(1, window)])
            incidences.append(incidence)
        else:
            incidences.append(prevalence)

    return incidences

df = pd.DataFrame({
    'years': y_years,
    'males_0_17': males_0_17["Patients"],
    'unknown_0_17': summed_other["Patients"],
    'females_0_17': females_0_17["Patients"],
    'males_12_17': males_12_17["Patients"],
    'females_12_17': females_12_17["Patients"],
    'gnrha_prevalence_12_17': total_12_17, 
    'gnrha_prevalence_0_17': total_0_17
    })

df["males_0_11"] = df["males_0_17"] - df["males_12_17"]
df["females_0_11"] = df["females_0_17"] - df["females_12_17"]
df["gnrha_prevalence_GD_12_17"] = df["gnrha_prevalence_12_17"]
cutoff_date = pd.Timestamp('2010-01-01')
df.loc[df.index < cutoff_date, 'gnrha_prevalence_GD_12_17'] = 0

df['3yr_duration_incidence'] = incidence_from_prevalence(df['gnrha_prevalence_GD_12_17'], 3)
df['4yr_duration_incidence'] = incidence_from_prevalence(df['gnrha_prevalence_GD_12_17'], 4)
df['5yr_duration_incidence'] = incidence_from_prevalence(df['gnrha_prevalence_GD_12_17'], 5)
df['cumsum_3'] = df['3yr_duration_incidence'].cumsum()
df['cumsum_4'] = df['4yr_duration_incidence'].cumsum()
df['cumsum_5'] = df['5yr_duration_incidence'].cumsum()
df["pop_12_17"] = pop_12_17.sum(axis=1)
df['pop_6_17'] = pop_0_17.loc[:, 6:17].sum(axis=1)
df['pop_period_start_2008'] = df.loc["2008-01-01", "pop_12_17"]
df['pop_period_start_2009'] = df.loc["2009-01-01", "pop_12_17"]
df['pop_period_start_2016'] = df.loc["2016-01-01", "pop_12_17"]
df['pop_period_start_2017'] = df.loc["2017-01-01", "pop_12_17"]
df['pop_period_start_6_17_2017'] = df.loc["2017-01-01", "pop_6_17"]

df['cum_3yr_inc_per_12_17_100k_2008'] = (df['cumsum_3'] / df['pop_period_start_2008']) * 100000
df['cum_3yr_inc_per_12_17_100k_2009'] = (df['cumsum_3'] / df['pop_period_start_2009']) * 100000
df['cum_4yr_inc_per_12_17_100k_2008'] = (df['cumsum_4'] / df['pop_period_start_2008']) * 100000
df['cum_4yr_inc_per_12_17_100k_2009'] = (df['cumsum_4'] / df['pop_period_start_2009']) * 100000


df['cumsum_3_2017_2021'] = df.loc['2016-01-01':'2020-01-01', '3yr_duration_incidence'].cumsum()

df['cum_3yr_inc_per_12_17_100k_2017'] = (df['cumsum_3_2017_2021'] / df['pop_period_start_2017']) * 100000


df.to_csv(f"./results/{name}.csv", float_format="%.2f")


print(df)









# ## Figures and Tables
# seaborn.set_theme()

# y_years = pd.to_datetime([f"{y}-01-01" for y in range(2006, 2023)])

# # https://www.engage.england.nhs.uk/consultation/puberty-suppressing-hormones/user_uploads/engagement-report-interim-policy-on-puberty-suppressing-hormones-for-gender-incongruence-or-dysphoria.pdf
# eng_wls = ([np.nan] * (2022 - 2006)) + [378] 
# hue = (["first"] * (2023 - 2006))





# df = pd.DataFrame({
#     'year': pd.Series(data=y_years, index=y_years),
#     # 'nz_12_17': f.loc[:, "Patients"] + m.loc[:, "Patients"],
#     'eng_wls_12_17': pd.Series(data=eng_wls, index=y_years),
#     "hue": pd.Series(data=hue, index=y_years),

#     # 'm_12_17': m.loc[:, "Patients"],
#                     },
#                     index=y_years)

# df.index.name = 'years'
# df.rename_axis("group", axis = "columns", inplace=True)


# ticks = [f"{x}-01-1" for x in range(2006, 2024, 2)]
# labels = [f"{x}" for x in range(2006, 2024, 2)]

# eng_wls_rate = ([np.nan] * (2022 - 2006)) + [9.6] 

# df2 = pd.DataFrame({
#     # 'nz_12_17_rate': (df1['nz_12_17'] / pop_12_17_.sum(axis=1)) * 100000,
#     'eng_wls_12_17_rate': pd.Series(data=eng_wls_rate, index=y_years),
#     }, index=y_years)





# colors = seaborn.color_palette("bright")
# colors.reverse()


# # # Fig 1
# seaborn.lineplot(x="year", y='nz_12_17', data=df1, color="#d62728", linestyle="solid", label="NZ adolescents 12-17")
# seaborn.lineplot(x="year", y='nz_9_11', data=df1, color="#ff7f0e", linestyle="dashed", label="NZ adolescents 9-11")
# seaborn.scatterplot(x="year", y='eng_wls_12_17', data=df1, label="England & Wales adolescents 12-17")

# # seaborn.lineplot(x="year", y='m_12_17', data=df1, color="#e31a1c", linestyle="dotted", label="Males 12-17")

# # plt.ylim([0, 450])
# # plt.yticks(ticks=[50, 100, 150, 200, 250, 300, 350, 400, 450])


# ## Rate
# # seaborn.lineplot(x="years", y='nz_12_17_rate', data=df2, color="#d62728", linestyle="solid", label="NZ adolescents 12-17 ")
# # seaborn.scatterplot(x="years", y='eng_wls_12_17_rate', data=df2, label="England & Wales adolescents 12-17")



# # # Fig 2
# # seaborn.lineplot(x="year", y='f_12_17', data=df4, color="#1f78b4", linestyle="dashed", label="Females 12-17")
# # seaborn.lineplot(x="year", y='f_10_11', data=df4, color="#1f78b4", linestyle="solid", label="Females 10-11")
# # seaborn.lineplot(x="year", y='f_0_9', data=df4, color="#1f78b4", linestyle="dotted", label="Females 0-9")
# # seaborn.lineplot(x="year", y='m_12_17', data=df4, color="#e31a1c", linestyle="dashed", label="Males 12-17")
# # seaborn.lineplot(x="year", y='m_10_11', data=df4, color="#e31a1c", linestyle="solid", label="Males 10-11")
# # seaborn.lineplot(x="year", y='m_0_9', data=df4, color="#e31a1c", linestyle="dotted", label="Males 0-9")
# # plt.yticks(ticks=[50, 100, 150, 200, 250])
# # plt.ylim([0, 250])

# plt.xticks(ticks=ticks, labels=labels)
# plt.rc('font', size=12)  
# plt.ylabel("Treatment prevalence")
# plt.xlabel(None)
# plt.xlim(["2006-01-01", "2023-01-01"])
# plt.legend(loc='upper left')






# # df1.style
# # dfi.export(df1, 'df_styled.png')


# plt.show()
