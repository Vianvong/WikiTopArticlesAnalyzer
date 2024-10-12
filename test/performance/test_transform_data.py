import numpy as np
import pandas as pd

from main.wiki_top_articles_analyzer import transform_data


def generate_large_dataframe(num_articles: int, start_date: str, end_date: str) -> pd.DataFrame:
    date_range = pd.date_range(start=start_date, end=end_date)
    num_dates = len(date_range)

    articles = [f"Article_{i}" for i in range(num_articles)]
    data = {
        "article": np.repeat(articles, num_dates),
        "date": date_range.tolist() * num_articles,
        "views": np.random.randint(100, 10000, size=num_articles * num_dates)
    }
    df = pd.DataFrame(data)
    return df


num_articles = 250000
start_date = "2023-01-01"
end_date = "2023-01-31"
df_test = generate_large_dataframe(num_articles, start_date, end_date)

transform_data(df_test)
