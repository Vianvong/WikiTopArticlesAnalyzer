import numpy as np
import pandas as pd

from main.wiki_top_articles_analyzer import calculate_stats


def generate_large_dataframe_for_stats(num_articles: int, num_rows: int) -> pd.DataFrame:
    articles = [f"Article_{i}" for i in range(num_articles)]
    data = {
        "article": np.random.choice(articles, size=num_rows),
        "views": np.random.randint(100, 10000, size=num_rows)
    }
    df = pd.DataFrame(data)

    num_nans = int(num_rows * 0.1)
    nan_indices = np.random.choice(df.index, size=num_nans, replace=False)
    df.loc[nan_indices, 'views'] = np.nan

    return df


num_articles = 100
num_rows = 100

df_test = generate_large_dataframe_for_stats(num_articles, num_rows)
mean_views, max_views = calculate_stats(df_test)
