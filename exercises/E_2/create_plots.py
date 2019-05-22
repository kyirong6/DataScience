import sys
import pandas as pd
import matplotlib.pyplot as plt


f1 = sys.argv[1]
f2 = sys.argv[2]

w1 = pd.read_csv(f1, sep=' ', header=None, index_col=1,
                 names=['lang', 'page', 'views', 'bytes'])
w2 = pd.read_csv(f2, sep=' ', header=None, index_col=1,
                 names=['lang', 'page', 'views', 'bytes'])
w1_views = w1.sort_values('views', ascending=False)['views']
w2_views = w2.sort_values('views', ascending=False)['views']
w1_views.name = "w1"
w2_views.name = "w2"
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(w1_views.values)
plt.title("Popularity Distribution")
plt.xlabel("Rank")
plt.ylabel("Views")
plt.subplot(1, 2, 2)
w1_w2 = pd.concat([w1_views, w2_views], axis=1).reset_index()
plt.scatter(w1_w2['w1'], w1_w2['w2'])
plt.xscale("log")
plt.yscale("log")
plt.title("Daily Correlation")
plt.xlabel("Day 1 views")
plt.ylabel("Day 2 views")
plt.savefig('wikipedia.png')
