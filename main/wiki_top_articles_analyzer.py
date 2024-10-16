import argparse
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame, to_datetime, concat, MultiIndex, date_range

API_BASE_URL = "https://wikimedia.org/api/rest_v1/metrics"
TOP_ENDPOINT = "pageviews/top"
TOP_ARGS = "{project}/{access}/{year}/{month}/{day}"
HEADERS = {"User-Agent": "wiki parser"}


def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{func.__name__} took {duration:.2f} seconds")
        return result

    return wrapper


def get_top_wiki_articles(project: str, year: str, month: str, day: str, access: str = "all-access") -> Optional[dict]:
    args = TOP_ARGS.format(project=project, access=access, year=year, month=month, day=day)
    return __api__(TOP_ENDPOINT, args)


def get_top_wiki_articles_async(project: str, year: str, month: str, day: str) -> Optional[DataFrame]:
    data = get_top_wiki_articles(project, year, month, day)
    if data and "items" in data and data["items"]:
        daily_df = DataFrame(data["items"][0]["articles"])
        daily_df["date"] = f"{year}-{month}-{day}"
        return daily_df
    return DataFrame()


def __api__(end_point: str, args: str, api_url: str = API_BASE_URL) -> Optional[dict]:
    url = "/".join([api_url, end_point, args])
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error receiving data from the API: {e}")
        return None


@timed
def process_dates(start: str, end: str) -> DataFrame:
    start_date = dt.datetime.strptime(start, "%Y%m%d")
    end_date = dt.datetime.strptime(end, "%Y%m%d")
    dates = [(str(d.year), str(d.month).zfill(2), str(d.day).zfill(2))
             for d in pd.date_range(start_date, end_date)]

    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(lambda d: get_top_wiki_articles_async("en.wikipedia", *d), dates))
    return concat(dfs, ignore_index=True)


@timed
def transform_data(df: DataFrame) -> DataFrame:
    df["date"] = to_datetime(df["date"])
    idx = MultiIndex.from_product(
        [df["article"].unique(), date_range(df["date"].min(), df["date"].max())],
        names=["article", "date"]
    )
    df = df.set_index(["article", "date"]).reindex(idx).groupby(level="article").ffill().reset_index()
    top_articles = df.groupby("article")["views"].last().nlargest(20).index
    return df[df["article"].isin(top_articles)]


@timed
def calculate_stats(df: DataFrame) -> Tuple[int, int]:
    views_sum = {article: 0 for article in df["article"].unique()}
    count = {article: 0 for article in df["article"].unique()}

    for _, row in df.iterrows():
        article = row["article"]
        views = row["views"]
        views_sum[article] += views
        count[article] += 1

    mean_views = {article: views_sum[article] / count[article] for article in views_sum}
    overall_mean_views = int(np.nanmean(list(mean_views.values())))
    max_views = df["views"].max()

    return overall_mean_views, max_views


@timed
def plot_data(df: DataFrame, mean_views: int, max_views: int, unique_articles: int) -> None:
    title = f"Top articles wiki views (Mean: {mean_views:.2f}, Max: {max_views}, Articles: {unique_articles})"
    plt.figure(figsize=(12, 8))

    for article, df_article in df.groupby("article"):
        plt.plot(df_article["date"], df_article["views"], label=article)

    plt.yscale("log")
    plt.title(title)
    plt.legend(loc='center')
    plt.savefig("top_articles.png")


def main():
    parser = argparse.ArgumentParser(description="Process start and end dates.")
    parser.add_argument("start", type=str, help="The start date in YYYY-MM-DD format")
    parser.add_argument("end", type=str, help="The end date in YYYY-MM-DD format")
    args = parser.parse_args()

    total_start_time = time.time()
    df = process_dates(args.start, args.end)
    unique_articles = df["article"].nunique()
    transformed_df = transform_data(df)
    mean_views, max_views = calculate_stats(transformed_df)
    plot_data(transformed_df, mean_views, max_views, unique_articles)
    total_end_time = time.time()

    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
