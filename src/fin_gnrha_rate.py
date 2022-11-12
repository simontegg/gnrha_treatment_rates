import pandas as pd

statistic = "gnrha_rate"
years = "2011_2018"
country = "fin"
name = f"{country}_{statistic}_{years}"

pop_source = '001_11s3_2021_20221112-060411.csv'
pop = pd.read_csv(f"./data/{pop_source}", skiprows=2, index_col=0)
pop = pop.drop('Area', axis=1)
pop = pop.drop('Urban-rural classification', axis=1)

total_pop_2011_9_17 = pop.iloc[0,9:18].sum()


upper_limit_2011_2018 = 186

print((upper_limit_2011_2018 / total_pop_2011_9_17) * 100000)

