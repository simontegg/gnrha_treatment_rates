import pandas as pd

statistic = "gnrha_n"
years = "2008_2021"
country = "eng_wls"
name = f"{country}_{statistic}_{years}"

source = 'butler_et_al_2022_tables_1_2.csv'
referred = pd.read_csv(f"./data/{source}") 

lt_18 = referred.query('age < 18')

started_gnrha = ["A1", "A3", "B2", "B3"]
possibly_started_gnrha = ["A4", "A6", "C2", "C3"]
did_not_start_gnrha = ["A2", "A5", "A7", "A8", "B1", "C1"]

started = started_gnrha + possibly_started_gnrha

cond = lt_18["category"].isin(started)
gnrha_lt_18 = lt_18[cond]

total = referred["n"].sum()
gnrha_lt_18_sum = gnrha_lt_18["n"].sum()

print(f"total referred to endocrinology: {total}")
print(f"total recieved gnrha and <18 at referral: {gnrha_lt_18_sum}")

