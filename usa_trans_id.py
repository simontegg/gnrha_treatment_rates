from datetime import date

import pandas as pd
import dataframe_image as dfi

# source_name = 'Household Pulse Survey PUF: October 5 - October 17'
# source_url = 'https://www.census.gov/programs-surveys/household-pulse-survey/datasets.html'
# source_start = '2022-10-05'
# source_end = '2022-10-17'
country = 'usa'
# source = 'pulse2022_puf_50.csv'
source = 'pulse2023_puf_63.csv'
year = 2023
statistic = "trans_id"
name = f"{country}_{statistic}_{year}"

usa = pd.read_csv(f"./data/{source}")


age_groups = [
        (18, 24),
        (25, 29),
        (30, 34),
        (35, 39),
        (40, 44),
        (45, 49),
        (50, 54),
        (55, 59),
        (60, 64),
        (65, 88), #1934 is the earliest birth year in HPS
        ]

income = {
        1: 12500,
        2: 30000,
        3: 42500,
        4: 62500,
        5: 87500,
        6: 125000,
        7: 175000,
        8: 210000,
        -99: False,
        -88: False
        }


def n(df):
    return df.shape[0]


index = []
stats = []
for age_group in age_groups:
    lower = age_group[0]
    upper = age_group[1]
    year_max = year - lower
    year_min = year - upper
    age_query = f"TBIRTH_YEAR >= {year_min} and TBIRTH_YEAR <= {year_max}"

    by_age =    usa.query(age_query)
    total =     n(by_age)

    
    males =     by_age.query("EGENID_BIRTH == 1")
    m_total =   n(males)
    m_f =       males.query("GENID_DESCRIBE == 2")
    m_trans =   males.query('GENID_DESCRIBE == 3')
    m_none =    males.query('GENID_DESCRIBE == 4')

    m_f_inc_3 = m_f.query('INCOME == 6')

    prop_all = n(males.query('INCOME == 6')) / m_total
    if n(m_f) > 0:
        prop = n(m_f_inc_3) / n(m_f)

        print('----')
        print(n(m_f))
        print(prop)
        print(prop_all)
    else:
        print('----')
        print(n(m_f))
        print('****')
        print(prop_all)


    females =   by_age.query("EGENID_BIRTH == 2")
    f_total =   n(females)
    f_m =       females.query("GENID_DESCRIBE == 1")
    f_trans =   females.query("GENID_DESCRIBE == 3")
    f_none =    females.query("GENID_DESCRIBE == 4")

    row = {
            "m_f": n(m_f) / m_total,
            "m_trans": n(m_trans) / m_total,
            "m_none": n(m_none) / m_total,
            "m_f_trans": (n(m_f) + n(m_trans)) / m_total,
            "m_trans_none": (n(m_trans) + n(m_none)) / m_total,
            "m_all": (n(m_f) + n(m_trans) + n(m_none)) / m_total,
            "f_m": n(f_m) / f_total,
            "f_trans": n(f_trans) / f_total,
            "f_none": n(f_none) / f_total,
            "f_m_trans": (n(f_m) + n(f_trans)) / f_total,
            "f_trans_none": (n(f_trans) + n(f_none)) / f_total,
            "f_all": (n(f_m) + n(f_trans) + n(f_none)) / f_total,
            "opposite_sex_id": (n(f_m) + n(m_f)) / total,
            "trans_nb": (n(f_m) + n(m_f) + n(m_trans) + n(f_trans) + n(m_none) + n(f_none)) / total,
            }

    index.append(f"{lower}_{upper}")
    stats.append(row)



df = pd.DataFrame(stats, index=index)
df.to_csv(f"./results/{name}.csv", float_format="%.4f")

# Table png
# percent = lambda x: "{:.2f}%".format(x*100)
# styled = df.style.format(percent)
# dfi.export(styled, f"results/{name}.png")






