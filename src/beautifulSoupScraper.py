from bs4 import BeautifulSoup
import requests
from src.iWebScraper import IWebScraper
import pandas as pd

class BeautifulSoupScraper(IWebScraper):
    def __init__(self, url):
        super().__init__(url)

    def fetch_html(self) -> str:
        req = requests.get(self.url)
        if req.status_code == 200:
            return req.text
        raise Exception(f"Failed to fetch HTML content. Status code: {req.status_code}")


    def parse(self, html_content: str) -> pd.DataFrame:
        soup = BeautifulSoup(html_content, 'html.parser')

        page_num = int(soup.select_one("li.current").get_text(strip=True).split()[-1]) if soup.select_one("li.current") else 1

        titles = []
        images = []
        descriptions = []
        for page in range(1, page_num + 1):
            page_soup = BeautifulSoup(requests.get(f"{self.url.replace('index.html', '')}page-{page}.html").text, 'html.parser')

            book_urls = list(map(lambda t: "https://books.toscrape.com/catalogue/" + t.get("href").split("../")[-1], page_soup.select("h3 a")))

            titles += list(map(lambda title: title.get("title"), page_soup.select("h3 a")))
            images += list(map(lambda image: "https://books.toscrape.com/" + image.get("src").split("../")[-1], page_soup.select("div a img")))

            descriptions += [BeautifulSoup(requests.get(url).text, 'html.parser').select_one("article.product_page > p").get_text(strip=True) for url in book_urls]

        categories = list(map(lambda category: category.get_text(strip=True), page_soup.select("div.page-header.action h1"))) * len(titles)

        return pd.DataFrame({
            "Title": titles,
            "Image": images,
            "Category": categories,
            "Description": descriptions
        })

        
    