import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
import requests

class DataCleaner():
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean(self) -> pd.DataFrame:
        print(f"Initial number of rows: {len(self.data)}")
        self.remove_missing_values()
        self.remove_duplicates()
        self.clean_text()
        self.remove_non_english()
        self.remove_invalid_url()

        self.data = self.data.reset_index(drop=True)

        print(f"Final number of rows: {len(self.data)}")
        return self.data

    def remove_missing_values(self):
        before = len(self.data)
        missing = self.data[["Title", "Category", "Description"]].isnull().sum()
        print(f"Missing values:\n{missing}")
        self.data = self.data.dropna(subset=["Title", "Category", "Description"])
        removed = before - len(self.data)
        print(f"Rows removed due to missing values: {removed}")

    def remove_duplicates(self):
        before = len(self.data)
        duplicates = self.data["Title"].duplicated().sum()
        print(f"Duplicate titles: {duplicates}")
        self.data = self.data.drop_duplicates(subset=["Title"])
        removed = before - len(self.data)
        print(f"Duplicate rows removed: {removed}")

    def clean_text(self):
        self.data["Title"] = self.data["Title"].str.replace(r'\s+', ' ', regex=True).str.strip()
        self.data["Description"] = self.data["Description"].str.replace(r'\s+', ' ', regex=True).str.strip()
        self.data["Category"] = self.data["Category"].str.replace(r'\s+', ' ', regex=True).str.strip()
        print("Text formatting cleaned.")

    def remove_non_english(self):
        before = len(self.data)
        non_english = self.analyze_non_english()
        percentage = non_english / len(self.data) * 100
        if percentage < 5:
            self.data = self.data[self.data["Description"].apply(self._is_english)]
            print(f"Non-English rows removed: {before - len(self.data)}")
        else:
            print("Non-English rows were not removed.")

    def remove_invalid_url(self):
        before = len(self.data)

        invalid_urls = self.analyze_invalid_urls()

        if invalid_urls > 0:
            self.data = self.data[
                self.data["Image"].apply(self._is_valid_url)
            ]
            print(f"Invalid image URLs removed: {before - len(self.data)}")
        else:
            print("No invalid image URLs found.")


    def analyze_invalid_urls(self):
        valid_count = self.data["Image"].apply(self._is_valid_url).sum()
        invalid_count = len(self.data) - valid_count

        print(f"Valid image URLs: {valid_count}, Invalid image URLs: {invalid_count}")

        return invalid_count
        
    def analyze_non_english(self):
        english_count = self.data['Description'].apply(self._is_english).sum()
        non_english_count = len(self.data) - english_count
        print(f"english: {english_count}, non-english: {non_english_count}")
        return non_english_count

    def _is_english(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return False
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False

    def _is_valid_url(self, url):
        if not isinstance(url, str) or not url.startswith('http'):
            return False
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            return (
                response.status_code == 200
                and 'image' in response.headers.get('Content-Type', '').lower()
            )
        except requests.RequestException:
            return False
