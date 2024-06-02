import polars as pl

statistic = "gnrha_rate"
years = "2011_2023"
country = "dnk"
name = f"{country}_{statistic}_{years}"

pop_source = 'FOLK2.csv'
pop = pl.read_csv(f"./data/{pop_source}", separator=";")

pop_2011 = pop.filter(
    (pop['ALDER'].is_between(12, 17)) &
    (pop['TID'] == 2011)
)

pop_2011_count = pop_2011["INDHOLD"].sum()


pop_2016 = pop.filter(
    (pop['ALDER'].is_between(12, 17)) &
    (pop['TID'] == 2016)
)

pop_2016_count = pop_2016["INDHOLD"].sum()

#Norup et al 2024
# https://doi.org/10.1210/clinem/dgae263
# note age ranges (10.9 - 18.0)
m_gnrha = 14
f_gnrha = 19
m_gnrha_subsq_csh = 29
f_gnrha_subsq_csh = 83
m_gnrha_and_csh = 12
f_gnrha_and_csh = 62

total_gnrha = m_gnrha + f_gnrha + m_gnrha_subsq_csh + f_gnrha_subsq_csh + m_gnrha_and_csh + f_gnrha_and_csh

print(total_gnrha)
print(pop_2011_count)
print(pop_2016_count)

print("Cumulative incidence from 2011-2023:Jan:")
print((total_gnrha / pop_2011_count) * 100_000)

print("Cumulative incidence from 2016-2023:Jan")
print((total_gnrha / pop_2016_count) * 100_000)


