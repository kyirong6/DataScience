import pandas as pd

totals = pd.read_csv('e1/totals.csv').set_index(keys=['name'])
counts = pd.read_csv('e1/counts.csv').set_index(keys=['name'])

lowest = totals.sum(1).idxmin(1)
avg = totals.sum(0).divide(counts.sum(0))
avg_city = totals.sum(1).divide(counts.sum(1))

print("City with lowest total precipitation:\n", lowest)
print("Average precipitation in each month:\n", avg)
print("Average precipitation in each city:\n", avg_city)
