import sys
import pandas as pd
from difflib import get_close_matches


def helper(title, data, tweets):
    lst = get_close_matches(title, tweets.title, n=len(tweets))
    if len(lst) == 0:
        return
    else:
        num = len(lst)
        overall_rating = sum(tweets[tweets['title'].isin(lst)].rating.values)
        data.at[data.index[data["title"] == title], 'rating'] = round(overall_rating/num, 2)


def main():
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    f3 = sys.argv[3]

    movie_data = pd.read_csv(f1, sep='\n', names=['title', 'rating'])
    twitter_data = pd.read_csv(f2, sep=',')
    twitter_data.rating = pd.to_numeric(twitter_data.rating.values)

    movie_data['title'].apply(helper, data=movie_data, tweets=twitter_data)

    movie_data = movie_data.dropna()
    movie_data = movie_data.sort_values('title')
    movie_data = movie_data.set_index('title')

    movie_data.to_csv(f3)


if __name__ == '__main__':
    main()



