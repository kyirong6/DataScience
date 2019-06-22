import pandas as pd
import scipy.stats as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main():
    data = pd.read_csv("data.csv")
    p_value = sp.f_oneway(data.qs1, data.qs2, data.qs3, data.qs4, data.merge1, data.partition_sort)[1]
    print(f"Anova p-value: {p_value}\n")

    data_melt = pd.melt(data)
    posthoc = pairwise_tukeyhsd(data_melt['value'], data_melt['variable'], alpha=0.05)
    fig = posthoc.plot_simultaneous()
    #fig.show()

    print(posthoc)


if __name__ == '__main__':
    main()

