from src.beautifulSoupScraper import BeautifulSoupScraper
from src.dataCleaner import DataCleaner
from src.dataAnalyser import DataAnalyser
from src.featureEngineer import FeatureEngineer
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

    train_data, test_data = train_test_split(
        cleaner.data,
        test_size=0.2,
        random_state=42,
        stratify=cleaner.data["Category"],
    )

    feature_engineer = FeatureEngineer()

    X_train = feature_engineer.fit_transform(train_data)
    X_test = feature_engineer.transform(test_data)

    y_train = train_data["Category"]
    y_test = test_data["Category"]

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    model = LogisticRegression(
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    results = test_data[["Title", "Category"]].copy()
    results["Predicted"] = y_pred

    wrong_predictions = results[
        results["Category"] != results["Predicted"]
    ]

    print("\nWrong predictions:")
    print(wrong_predictions.to_string(index=False))

    print("=" * 40)
    print("CONFUSION MATRIX")
    print("=" * 40)
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()