from datetime import date

import pandas as pd
import dataframe_image as dfi

country = "nld"
pop_source = "7461eng_TypedDataSet_29102022_160841.csv"
gnrha_source = "table_1_wiepjes_et_al_2018.csv"
statistic = "gnrha_rate"
years = "1987_2015"
name = f"{country}_{statistic}_{years}"

male = "3000   "
female = "4000   "
pop = 'TotalPopulation_1'
numeric_cols = ["transwomen", "transmen", "total"]
ages = [
        # 10900, #9 years
        # 11000, #10 years
        # 11100, #11 years
        11200, #12 years
        11300, #13 years
        11400, #14 years
        11500, #15 years
        11600, #16 years
        11700, #17 years
        ]

def year_codes(start, stop):
    return [f"{y}JJ00" for y in range(start, stop)]

table_1 = pd.read_csv(f"./data/{gnrha_source}", index_col="category")
table_1[numeric_cols] = table_1[numeric_cols].apply(pd.to_numeric)

# compute numbers given gnrha
table_1.loc['adolescents_ps_n'] = (
        (table_1.iloc[11] * table_1.iloc[7])
        ).apply(lambda x: round(x))

table_1.loc['children_ps_n'] = (
        (table_1.iloc[15] * table_1.iloc[19])
        ).apply(lambda x: round(x))

table_1.loc['minors_ps_n'] = table_1.iloc[21] + table_1.iloc[22]


table_1.loc['minors_ps_ratio'] = table_1.iloc[23] / table_1.at['minors_ps_n', 'total']

# 118 treated with gnrha 1987-2008 according to Biggs (2022)
# subtract this from the total to give number treated since 2009
# assume the sex ratio was even based on De Vries et al (2011)
table_1.loc['1987_2008'] = [59, 59, 118]
table_1.loc['2009_2015'] = table_1.loc['minors_ps_n'] - table_1.loc['1987_2008']

total_minors_gnrha = table_1.at['minors_ps_n', 'total']
f_minors_gnrha = table_1.at['minors_ps_n', 'transmen']
m_minors_gnrha = table_1.at['minors_ps_n', 'transwomen']

total_minors_gnrha_2009_2015 = table_1.at['2009_2015', 'total']
f_minors_gnrha_2009_2015 = table_1.at['2009_2015', 'transmen']
m_minors_gnrha_2009_2015 = table_1.at['2009_2015', 'transwomen']


total_minors_gnrha_2009_2018 = 720 + 229 - 118
f_minors_gnrha_2009_2018 = int((500/720) * total_minors_gnrha_2009_2018)
m_minors_gnrha_2009_2018 = int((220/720) * total_minors_gnrha_2009_2018)

# compute child and adolescent population through years 1987-2015
population = pd.read_csv(f"./data/{pop_source}", sep=";")
year_ranges = [
        #start, end, cumulative
        (1987, 2015, False),
        (2009, 2015, False),
        (1987, 2015, True),
        (2009, 2015, True),
        (2009, 2018, False),
        (2009, 2018, True),
        ]

index = []
stats = []
formatter = {}
for r in year_ranges:
    # use a year_codes for yearly mean prevalence
    # years = year_codes(r[0], r[1]) 
    
    # use the period start as the denominator for cumulative incidence
    years = [f"{r[0]}JJ00"] if r[2] else year_codes(r[0], r[1])

    cohort = population[population.Periods.isin(years) & population.Age.isin(ages)]
    males = cohort.query(f"Sex == '{male}'")
    females = cohort.query(f"Sex == '{female}'")
    m_sum = males[pop].sum()
    f_sum = females[pop].sum()
    total_sum = m_sum + f_sum

    row = {}
    row["total"] = total_sum
    row["female"] = f_sum
    row["male"] = m_sum
    row_name = f"9_17_pop_{r[0]}_{r[1]}" if len(years) > 1 else f"9_17_pop_{r[0]}"
    index.append(row_name)
    stats.append(row)
    formatter[row_name] = "{:,.0f}"

    gnrha_row = {}
    if r[0] == 1987:
        gnrha_row['total'] = total_minors_gnrha
        gnrha_row['female'] = f_minors_gnrha
        gnrha_row['male'] = m_minors_gnrha

    if r[0] == 2009 and r[1] == 2015:
        gnrha_row['total'] = total_minors_gnrha_2009_2015
        gnrha_row['female'] = f_minors_gnrha_2009_2015
        gnrha_row['male'] = m_minors_gnrha_2009_2015

    if r[0] == 2009 and r[1] == 2018:
        gnrha_row['total'] = total_minors_gnrha_2009_2018
        gnrha_row['female'] = f_minors_gnrha_2009_2018
        gnrha_row['male'] = m_minors_gnrha_2009_2018

    stats.append(gnrha_row)
    gnrha_row_name = f"gnrha_n_{r[0]}_{r[1]}"
    index.append(gnrha_row_name)
    formatter[gnrha_row_name] = "{:,.0f}"

    rate = {}
    rate['total'] = (gnrha_row['total'] / total_sum) * 100000
    rate['female'] = (gnrha_row['female'] / f_sum) * 100000
    rate['male'] = (gnrha_row['male'] / m_sum) * 100000
    stats.append(rate)
    rate_name = f"yearly_mean_gnrha_per_100k_{r[0]}_{r[1]}" if len(years) > 1 else f"cumulative_gnrha_per_100k_{r[0]}_{r[1]}"
    index.append(rate_name)
    formatter[rate_name] = "{:.2f}"


df = pd.DataFrame(stats, index=index)
df = df.transpose()
df.to_csv(f"./results/{name}.csv", float_format="%.2f")

styled = df.style.format(formatter=formatter)
dfi.export(styled, f"results/{name}.png")














