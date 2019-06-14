import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def is_weekend(x):
    return x.weekday() in [5, 6]


def is_weekday(x):
    return x.weekday() not in [5, 6]


def get_year(x):
    return x.isocalendar()[0]


def get_week(x):
    return x.isocalendar()[1]


def get_year_week(x):
    return x.isocalendar()[0], x.isocalendar()[1]


def main():
    # f1 = sys.argv[1]
    counts = pd.read_json("reddit-counts.json.gz", lines=True)
    counts_weekday = counts.copy(deep=True)
    counts_weekly_weekend = counts.copy(deep=True)
    counts_weekly_weekday = counts.copy(deep=True)
    counts_mann_weekend = counts.copy(deep=True)
    counts_mann_weekday = counts.copy(deep=True)

    counts = counts[counts["subreddit"] == "canada"]
    counts["day_numbers"] = counts["date"].apply(is_weekend)
    counts = counts[counts["day_numbers"] == 1]
    del counts["day_numbers"]
    counts["year"] = counts["date"].dt.year
    counts = counts[counts["year"].isin([2012, 2013])]
    counts_weekend = counts

    counts_weekday = counts_weekday[counts_weekday["subreddit"] == "canada"]
    counts_weekday["day_numbers"] = counts_weekday["date"].apply(is_weekday)
    counts_weekday = counts_weekday[counts_weekday["day_numbers"] == 1]
    del counts_weekday["day_numbers"]
    counts_weekday["year"] = counts_weekday["date"].dt.year
    counts_weekday = counts_weekday[counts_weekday["year"].isin([2012, 2013])]


    # ------t test--------
    p_val = stats.ttest_ind(counts_weekend["comment_count"], counts_weekday["comment_count"])[1]
    print(f"Initial (invalid) T-test p-value: {p_val}\n")

    # ------normality test------
    p_norm_weekend = stats.normaltest(counts_weekend["comment_count"]).pvalue
    p_norm_weekday = stats.normaltest(counts_weekday["comment_count"]).pvalue
    print(f"Original data normality p-values: weekend: {p_norm_weekend:.3g} weekday: {p_norm_weekday:.3g}\n")

    # ------levene test---------
    initial_levene_p = stats.levene(counts_weekend["comment_count"], counts_weekday["comment_count"])[1]
    print(f"Original data equal-variance p-value: {initial_levene_p:.3g}\n")

    # ---------fix 1------------
    counts_weekend["comment_count"] = counts_weekend["comment_count"].transform(np.sqrt)
    # plt.hist(counts_weekend["comment_count"])
    # plt.show()
    counts_weekday["comment_count"] = counts_weekday["comment_count"].transform(np.sqrt)
    # plt.hist(counts_weekday["comment_count"])
    # plt.show()
    transformed_weekday_normality_p = stats.normaltest(counts_weekday["comment_count"]).pvalue
    transformed_weekend_normality_p = stats.normaltest(counts_weekend["comment_count"]).pvalue
    print(f"Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n")
    transformed_levene_p = stats.levene(counts_weekend["comment_count"], counts_weekday["comment_count"])[1]
    print(f"Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n")

    # ----------fix2-------------
    counts_weekly_weekend = counts_weekly_weekend[counts_weekly_weekend["subreddit"] == "canada"]
    counts_weekly_weekend["day_numbers"] = counts_weekly_weekend["date"].apply(is_weekend)
    counts_weekly_weekend = counts_weekly_weekend[counts_weekly_weekend["day_numbers"] == 1]
    del counts_weekly_weekend["day_numbers"]
    counts_weekly_weekend["year"] = counts_weekly_weekday["date"].dt.year
    counts_weekly_weekend = counts_weekly_weekend[counts_weekly_weekend["year"].isin([2012, 2013])]

    counts_weekly_weekday = counts_weekly_weekday[counts_weekly_weekday["subreddit"] == "canada"]
    counts_weekly_weekday["day_numbers"] = counts_weekly_weekday["date"].apply(is_weekday)
    counts_weekly_weekday = counts_weekly_weekday[counts_weekly_weekday["day_numbers"] == 1]
    del counts_weekly_weekday["day_numbers"]
    counts_weekly_weekday["year"] = counts_weekly_weekday["date"].dt.year
    counts_weekly_weekday = counts_weekly_weekday[counts_weekly_weekday["year"].isin([2012, 2013])]

    counts_weekly_weekend["year_week"] = counts_weekly_weekend["date"].apply(get_year_week)
    counts_weekly_weekday["year_week"] = counts_weekly_weekday["date"].apply(get_year_week)

    counts_weekly_weekend = counts_weekly_weekend.groupby(["year_week"]).aggregate({"comment_count": "mean"})
    counts_weekly_weekday = counts_weekly_weekday.groupby(["year_week"]).aggregate({"comment_count": "mean"})
    counts_weekly_weekday.drop(counts_weekly_weekday.tail(1).index, inplace=True)
    print(counts_weekly_weekday.describe())
    print(counts_weekly_weekend.describe())

    weekly_weekend_normality_p = stats.normaltest(counts_weekly_weekend["comment_count"]).pvalue
    weekly_weekday_normality_p = stats.normaltest(counts_weekly_weekday["comment_count"]).pvalue
    weekly_ttest_p = stats.ttest_ind(counts_weekly_weekday["comment_count"], counts_weekly_weekend["comment_count"])[1]
    weekly_levene_p = stats.levene(counts_weekly_weekday["comment_count"], counts_weekly_weekend["comment_count"])[1]
    print(f"Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n")
    print(f"Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n")
    print(f"Weekly T-test p-value: {weekly_ttest_p:.3g}\n")

    # -----------fix3-----------------
    counts_mann_weekend = counts_mann_weekend[counts_mann_weekend["subreddit"] == "canada"]
    counts_mann_weekday = counts_mann_weekday[counts_mann_weekday["subreddit"] == "canada"]

    counts_mann_weekend["day_numbers"] = counts_mann_weekend["date"].apply(is_weekend)
    counts_mann_weekend = counts_mann_weekend[counts_mann_weekend["day_numbers"] == 1]
    del counts_mann_weekend["day_numbers"]
    counts_mann_weekend["year"] = counts_mann_weekend["date"].dt.year
    counts_mann_weekend = counts_mann_weekend[counts_mann_weekend["year"].isin([2012, 2013])]

    counts_mann_weekday["day_numbers"] = counts_mann_weekday["date"].apply(is_weekday)
    counts_mann_weekday = counts_mann_weekday[counts_mann_weekday["day_numbers"] == 1]
    del counts_mann_weekday["day_numbers"]
    counts_mann_weekday["year"] = counts_mann_weekday["date"].dt.year
    counts_mann_weekday = counts_mann_weekday[counts_mann_weekday["year"].isin([2012, 2013])]

    utest_p = stats.mannwhitneyu(counts_mann_weekend["comment_count"], counts_mann_weekday["comment_count"], alternative="two-sided")[1]
    print(f"Mann–Whitney U-test p-value: {utest_p:.3g}")


if __name__ == '__main__':
    main()

