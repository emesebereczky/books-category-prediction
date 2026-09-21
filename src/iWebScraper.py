from abc import ABC, abstractmethod
import pandas as pd

class IWebScraper(ABC):
    def __init__(self, url):
        self.url = url

    @abstractmethod
    def fetch_html(self) -> str:
        pass

    @abstractmethod
    def parse(self, html_content: str) -> pd.DataFrame:
        pass

    def scrape(self) -> pd.DataFrame:
        html = self.fetch_html()
        return self.parse(html)
