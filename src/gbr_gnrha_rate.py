import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi
import seaborn

import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

statistic = "gnrha_rate"
years = "2008_2014_2018"
country = "eng_wls"
name = f"{country}_{statistic}_{years}"

pop_source = 'ons_mye_custom_age_tool_2020.xlsx'
pop = pd.read_excel(f"./data/{pop_source}", skiprows=1, sheet_name="Single year of age", header=[0, 1], index_col=[1, 2], convert_float=False) 

pop.index.names = ["year", "borough"]
pop.columns.names = ["sex", "age"]
eng_wls = pop.query("borough == 'England and Wales'")
eng_wls.sort_index(axis=1, inplace=True, ascending=True) 
eng_wls.sort_index(inplace=True, ascending=True) 

# select ages 9-14 between 2014-2018
males = eng_wls.loc[
        (2014, "England and Wales"):(2018, "England and Wales"), 
        ("M", 9):("M", 14)
        ]
females = eng_wls.loc[
        (2014, "England and Wales"):(2018, "England Wales"), 
        ("F", 9):("F", 14)
        ]

# sum across ages 9-14
males['total'] = males.sum(axis=1)
females['total'] = females.sum(axis=1)
total = males['total'] + females['total'] 



gnrha_yearly_source = '18-19333_tavistock_nhs.csv'
gnrha = pd.read_csv(f"./data/{gnrha_yearly_source}")

# sum current incidence with previous 2 years to compute estimated prevalence
gnrha['prevalence'] = gnrha['gnrha_referrals_lt_15'].rolling(
        min_periods=3, 
        window=3
        ).sum()

df = gnrha[gnrha['year'] > 2013]
df['pop_9_14'] = total.tolist()

df['gnrha_per_100k_9_14'] = (df['prevalence'] / df['pop_9_14']) * 100000
df['year'] = pd.date_range(start="2014-01-01", end="2018-01-01", freq="AS")

seaborn.set(style="ticks")

fig, ax = plt.subplots()

ax.plot(df['year'],
       df['gnrha_per_100k_9_14'],
       color='blue')

# Set title and labels for axes
ax.set(xlabel="Year",
       ylabel="GnRha treatment rate per 100k",
       title="Children 9-14 - England and Wales 2014-2018")


ax.set_ylim([0, 20])

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())

formatter = {
        'year': lambda t: t.strftime("%Y"),
        'gids_referrals_lt_16': "{:,.0f}",
        'gnrha_referral_lt_16': "{:,.0f}",
        'prevalence': "{:,.0f}",
        'pop_9_14': "{:,.0f}",
        'gnrha_per_100k_9_14': "{:.2f}",
        }

# plt.show()

plt.savefig(f"./results/{name}_chart.png")

## TODO full figures with 2012, 2013
df.to_csv(f"./results/{name}.csv", float_format="%.2f")

styled = df.style.format(formatter=formatter)
dfi.export(styled, f"results/{name}_table.png")

print(eng_wls)

males_12_17 = eng_wls.loc[
        (2008, "England and Wales"):(2020, "England and Wales"), 
        ("M", 12):("M", 17)
        ]

females_12_17 = eng_wls.loc[
        (2008, "England and Wales"):(2020, "England Wales"), 
        ("F", 12):("F", 17)
        ]



pop_m_2008 = males_12_17.loc[(2008, "England and Wales"), :].sum()
pop_f_2008 = females_12_17.loc[(2008, "England and Wales"), :].sum()
pop_total_2008 = pop_m_2008 + pop_f_2008

# pop_m_2020 = males_12_17.loc[(2020, "England and Wales"), :].sum()
# pop_f_2020 = females_12_17.loc[(2020, "England and Wales"), :].sum()
# pop_total_2020 = pop_m_2020 + pop_f_2020

# print((378 / pop_total_2020) * 100000)

cum_df = pd.DataFrame({

    'pop_2008_12_17': [pop_total_2008, pop_f_2008, pop_m_2008],
    'gnrha_gd_2008_2020_butler_cass': [851 + 727, None, None],
    'cum_rate_per_12_17_100k_butler_cass': [((851 + 727) / pop_total_2008) * 100000, None, None],
    'gnrha_gd_2008_2019_butler_cass': [1600, None, None],
    'cum_rate_per_12_17_100k_butler_cass': [(1600 / pop_total_2008) * 100000, None, None],
    'gnrha_gd_2008_2020_barnes': [2000, None, None],
    'cum_rate_per_12_17_100k__barnes': [(2000 / pop_total_2008) * 100000, None, None],
    'gnrha_gd_2008_2020_high': [2500, None, None],
    'cum_rate_per_12_17_100k_high': [(2500 / pop_total_2008) * 100000, None, None],
    },
    index=["total", "female", "male"])

cum_df.to_csv(f"./results/{name}_cumulative.csv", float_format="%.2f")

# print(cum_df)



# females_10_19 = eng_wls.loc[
#         (2020, "England and Wales"), 
#         ("F", 10):("F", 19)
#         ]

# males_10_19 = eng_wls.loc[
#         (2020, "England and Wales"), 
#         ("M", 10):("M", 19)
#         ]

# females_10_19_sum = females_10_19.sum()
# males_10_19_sum = males_10_19.sum()



# total_2020_10_19 = females_10_19_sum + males_10_19_sum

# ONS https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationandhouseholdestimatesenglandandwalescensus2021
# pop_10_14_2021 = 3595900
# pop_15_19_2021 = 3394700
# total_2021_10_19 = pop_10_14_2021 + pop_15_19_2021
# multiplier = total_2021_10_19 / total_2020_10_19

# print(total_2020_10_19)
# print(total_2021_10_19)
# print(multiplier)




















# eng_wls_pop = gbr_pop.loc[gbr_pop.columns.get_level_values(0) == "M"]


