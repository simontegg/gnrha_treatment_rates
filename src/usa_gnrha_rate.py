import pandas as pd

statistic = "gnrha_rate"
years = "2017_2021"
country = "usa"
name = f"{country}_{statistic}_{years}"

pop_source = 'nc-est2019-syasexn.xlsx'
pop = pd.read_excel(f"./data/{pop_source}", skiprows=3, index_col=0, skipfooter=215)

pop = pop.drop('Total\nPopulation')
pop.index = range(0, 101)

pop_2017_9_17 = pop.loc[9:17, 2017].sum()
pop_2017_6_17 = pop.loc[6:17, 2017].sum()
pop_2019_12_17 = pop.loc[12:17, 2019].sum()

# https://www.reuters.com/investigates/special-report/usa-transyouth-data/
cumulative_incidence_2017_2021 = 4780
cumulative_incidence_2019_2023 = 8579

df = pd.DataFrame({
    'cumulative_incidence_2017_2021': cumulative_incidence_2017_2021,
    'pop_2017_6_17': pop_2017_6_17,
    'pop_2017_9_17': pop_2017_9_17,
    'pop_2019_12_17': pop_2019_12_17,
    'gnrha_rate_9_17_100k': (cumulative_incidence_2017_2021 / pop_2017_9_17) * 100000,
    'gnrha_rate_6_17_100k': (cumulative_incidence_2017_2021 / pop_2017_6_17) * 100000,
    'gnrha_rate_2019_2023_12_17_100k': (cumulative_incidence_2019_2023 / pop_2019_12_17) * 100000
    }, index=["total"])

df.to_csv(f"./results/{name}.csv", float_format="%.2f")

print(df)

