from datetime import date
from matplotlib.pyplot import axis

import pandas as pd
import dataframe_image as dfi

# source_name = 'Household Pulse Survey PUF: October 5 - October 17'
# source_url = 'https://www.census.gov/programs-surveys/household-pulse-survey/datasets.html'
# source_start = '2022-10-05'
# source_end = '2022-10-17'
country = 'usa'
source = 'pulse2022_puf_50.csv'
year = 2022
statistic = "trans_id"
name = f"{country}_{statistic}_{year}"

usa = pd.read_csv(f"./data/{source}")

age_groups = [
        (18, 24)
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

def get_mean_income(df):
    count = 0
    total = 0
    for i in range(1, 8):
        income_bracket = n(df.query(f"INCOME == {i}")) 
        count += income_bracket
        print(f"number in income bracket {i}: {income_bracket}")
        total += income_bracket * income[i]

    return total / count

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
    m_f =       males.query("GENID_DESCRIBE == 2 or GENID_DESCRIBE == 3")
    m_trans =   males.query('GENID_DESCRIBE == 3')
    m_none =    males.query('GENID_DESCRIBE == 4')

    females =   by_age.query("EGENID_BIRTH == 2")
    f_total =   n(females)
    f_m =       females.query("GENID_DESCRIBE == 1 or GENID_DESCRIBE == 3")
    f_trans =   females.query("GENID_DESCRIBE == 3")
    f_none =    females.query("GENID_DESCRIBE == 4")

    m_f_inc = get_mean_income(m_f)
    m_inc = get_mean_income(males)

    f_m_inc = get_mean_income(f_m)
    f_inc = get_mean_income(females)

    m_m = males.query("GENID_DESCRIBE == 1")
    male_ls_tim_pls_tif = pd.concat([m_m, f_m], axis=0)
    gender_male_inc = get_mean_income(male_ls_tim_pls_tif)

    f_f = females.query("GENID_DESCRIBE == 2")
    gender_female = pd.concat([f_f, m_f], axis=0)
    gender_female_inc = get_mean_income(gender_female)

    
    print(f"tim income: {m_f_inc}")
    print(f"overall male income: {m_inc}")
    print(f"gender male income: {gender_male_inc}")

    print(f"tif income: {f_m_inc}")
    print(f"female income: {f_inc}")
    print(f"gender female income: {gender_female_inc}")


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



# df = pd.DataFrame(stats, index=index)
# df.to_csv(f"./results/{name}.csv", float_format="%.4f")

# Table png
# percent = lambda x: "{:.2f}%".format(x*100)
# styled = df.style.format(percent)
# dfi.export(styled, f"results/{name}.png")






