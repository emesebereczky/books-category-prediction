from src.beautifulSoupScraper import BeautifulSoupScraper
from src.dataCleaner import DataCleaner
import pandas as pd

def main():
    test = BeautifulSoupScraper("https://books.toscrape.com/catalogue/category/books/fiction_10/index.html")
    test2 = BeautifulSoupScraper("https://books.toscrape.com/catalogue/category/books/nonfiction_13/index.html")
    data: pd.DataFrame = pd.concat([test.scrape(), test2.scrape()], ignore_index=True)
    print(data)
    cleaner = DataCleaner(data)
    cleaner.clean()

if __name__ == "__main__":
    main()