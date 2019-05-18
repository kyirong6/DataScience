import numpy as np

data = np.load('e1/monthdata.npz')
totals = data['totals']
counts = data['counts']

lowest = np.argmin(np.sum(totals, 1))
avg = np.divide(np.sum(totals, 0), np.sum(counts, 0))
avg_city = np.divide(np.sum(totals, 1), np.sum(counts, 1))
qrter = np.reshape(totals, (-1, 4, 3)).sum(axis=2)

print("Row with lowest total precipitation:\n", lowest)
print("Average precipitation in each month:\n", avg)
print("Average precipitation in each city:\n", avg_city)
print("Quarterly precipitation totals:\n", qrter)
