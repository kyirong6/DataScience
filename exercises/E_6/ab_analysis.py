import sys
import pandas as pd
import numpy as np
import scipy.stats as sp


def main():
    f1 = sys.argv[1]

    OUTPUT_TEMPLATE = (
        '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
        '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
        '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
        '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
    )

    searches = pd.read_json(f1, orient='records', lines=True)
    odd_uid = searches[searches["uid"] % 2 == 1]
    even_uid = searches[searches["uid"] % 2 == 0]
    odd_uid_instr = odd_uid[odd_uid["is_instructor"] == 1]
    even_uid_instr = even_uid[even_uid["is_instructor"] == 1]

    val1 = odd_uid[odd_uid["search_count"] > 0].count()[1]
    val2 = odd_uid[odd_uid["search_count"] == 0].count()[1]
    val3 = even_uid[even_uid["search_count"] > 0].count()[1]
    val4 = even_uid[even_uid["search_count"] == 0].count()[1]
    data_cont = np.array([[val1, val2], [val3, val4]])
    more_users_p = sp.chi2_contingency(data_cont)[1]

    val11 = odd_uid_instr[odd_uid_instr["search_count"] > 0].count()[1]
    val22 = odd_uid_instr[odd_uid_instr["search_count"] == 0].count()[1]
    val33 = even_uid_instr[even_uid_instr["search_count"] > 0].count()[1]
    val44 = even_uid_instr[even_uid_instr["search_count"] == 0].count()[1]
    data_cont2 = np.array([[val11, val22], [val33, val44]])
    more_instr_p = sp.chi2_contingency(data_cont2)[1]

    more_searches_p = sp.mannwhitneyu(odd_uid["search_count"], even_uid["search_count"], alternative='two-sided').pvalue
    more_instr_searches_p = sp.mannwhitneyu(odd_uid_instr["search_count"], even_uid_instr["search_count"], alternative='two-sided').pvalue

    print(OUTPUT_TEMPLATE.format(
        more_users_p=more_users_p,
        more_searches_p=more_searches_p,
        more_instr_p=more_instr_p,
        more_instr_searches_p=more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()
