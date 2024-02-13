import pandas as pd

statistic = "gnrha_rate"
years = "2011_2021"
country = "che"
name = f"{country}_{statistic}_{years}"

## converted to csv from './data/px-x-0102030000_101.px'
pop_source = 'swiss_population.csv'
pop = pd.read_csv(f"./data/{pop_source}")
pop['Alter'] = pop['Alter'].str.replace(r'\D', '', regex=True)
pop['Alter'] = pd.to_numeric(pop['Alter'], errors='coerce').fillna(0).astype(int)

males = pop.query("Geschlecht == 'Mann'")
females = pop.query("Geschlecht == 'Frau'")
m_12_17 = males.query("Alter > 11 & Alter < 18 & Jahr == 2016")
f_12_17 = females.query("Alter > 11 & Alter < 18 & Jahr == 2016")

male_12_17_2016 = m_12_17["value"].sum()
female_12_17_2016 = f_12_17["value"].sum()

pop_12_17_2016 = male_12_17_2016 + female_12_17_2016

# https://www.gavinpublishers.com/article/view/pediatric-transgender-care-experience-of-a-swiss-tertiary-center-over-the-past-decade
# median 25 at 9 centres
# Assume mean of 30 = 270 
# + 27 = 297 = ~300
gnrha_treated_2016_2021 = 300
cumulative_incidence_per_100k = (gnrha_treated_2016_2021 / pop_12_17_2016) * 100000

print(cumulative_incidence_per_100k)






