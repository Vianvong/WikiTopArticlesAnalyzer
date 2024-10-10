import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from main import wiki_top_articles_analyzer


class TestWikiParser(unittest.TestCase):
    @patch('main.wiki_top_articles_analyzer.requests.get')
    def test_get_top_wiki_articles(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "items": [{"articles": [{"article": "Python", "views": 1000}]}]
        }
        result = wiki_top_articles_analyzer.get_top_wiki_articles("en.wikipedia", "2024", "01", "01")
        self.assertEqual(result["items"][0]["articles"][0]["article"], "Python")

    def test_transform_data(self):
        data = {"article": ["Python", "Java"], "views": [1000, 1500], "date": ["2024-01-01", "2024-01-01"]}
        df = pd.DataFrame(data)
        result = wiki_top_articles_analyzer.transform_data(df)
        self.assertEqual(len(result["article"].unique()), 2)
        self.assertTrue("views" in result.columns)

    def test_calculate_stats(self):
        data = {"article": ["Python", "Java"], "views": [1000, 1500], "date": ["2024-01-01", "2024-01-01"]}
        df = pd.DataFrame(data)
        mean_views, max_views, unique_articles = wiki_top_articles_analyzer.calculate_stats(df)
        self.assertEqual(mean_views, 1250)
        self.assertEqual(max_views, 1500)
        self.assertEqual(unique_articles, 2)


if __name__ == "__main__":
    unittest.main()
