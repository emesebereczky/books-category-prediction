from src.beautifulSoupScraper import BeautifulSoupScraper
from src.dataCleaner import DataCleaner
from src.dataAnalyser import DataAnalyser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    test = BeautifulSoupScraper("https://books.toscrape.com/catalogue/category/books/fiction_10/index.html")
    test2 = BeautifulSoupScraper("https://books.toscrape.com/catalogue/category/books/nonfiction_13/index.html")
    data: pd.DataFrame = pd.concat([test.scrape(), test2.scrape()], ignore_index=True)
    print(data)
    cleaner = DataCleaner(data)
    cleaner.clean()

    analyzer = DataAnalyser(cleaner.data)
    analyzer.analyze()

"""
    train_data: pd.DataFrame
    test_data: pd.DataFrame

    train_data, test_data = train_test_split(
        cleaner.data,
        test_size=0.2,
        random_state=42,
        stratify=cleaner.data["Category"],
    )
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    train_text = (
        train_data["Title"].fillna("") + " " + train_data["Description"].fillna("")
    )
    test_text = (
        test_data["Title"].fillna("") + " " + test_data["Description"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=300,
        ngram_range=(1, 1),
        min_df=2
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    y_train = train_data["Category"]
    y_test = test_data["Category"]

    model = LogisticRegression(class_weight="balanced", C=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 40)
    print("CLASSIFICATION REPORT")
    print("=" * 40)
    print(classification_report(y_test, y_pred))

    print("=" * 40)
    print("CONFUSION MATRIX")
    print("=" * 40)
    print(confusion_matrix(y_test, y_pred))
"""


if __name__ == "__main__":
    main()