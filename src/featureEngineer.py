import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.imageFeatureExtractor import ImageFeatureExtractor


class FeatureEngineer:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=200,
            min_df=2
        )

        self.scaler = StandardScaler()
        self.image_extractor = ImageFeatureExtractor()

    def create_text(self, data: pd.DataFrame) -> pd.Series:
        return (
            data["Title"].fillna("")
            + " "
            + data["Description"].fillna("")
        )

    def create_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:

        features = pd.DataFrame(index=data.index)

        features["title_length"] = data["Title"].fillna("").str.len()
        features["description_length"] = (
            data["Description"].fillna("").str.len()
        )

        features["title_word_count"] = (
            data["Title"]
            .fillna("")
            .str.split()
            .str.len()
        )

        features["description_word_count"] = (
            data["Description"]
            .fillna("")
            .str.split()
            .str.len()
        )

        #?????
        features["title_description_ratio"] = (
            features["title_length"]
            / features["description_length"].replace(0, 1)
        )

        image_results = data["Image"].apply(
            self.image_extractor.extract
        )

        image_features_df = pd.DataFrame(
            image_results.tolist(),
            index=data.index
        )

        features = pd.concat(
            [features, image_features_df],
            axis=1
        )

        print("\nNumeric features:")
        print(features.head())

        return features

    def fit_transform(self, data: pd.DataFrame):

        text = self.create_text(data)

        X_text = self.vectorizer.fit_transform(text)

        print("\nTF-IDF features:")
        print("Number of TF-IDF features:", len(self.vectorizer.get_feature_names_out()))

        print(
            self.vectorizer.get_feature_names_out()[:20]
        )

        numeric_features = self.create_numeric_features(data)

        X_numeric = self.scaler.fit_transform(numeric_features)

        X_numeric = csr_matrix(X_numeric)

        X = hstack([X_text, X_numeric])

        print("\nFinal feature matrix:")
        print("Shape:", X.shape)

        return X

    def transform(self, data: pd.DataFrame):

        text = self.create_text(data)

        X_text = self.vectorizer.transform(text)

        numeric_features = self.create_numeric_features(data)

        X_numeric = self.scaler.transform(numeric_features)
        X_numeric = csr_matrix(X_numeric)

        X = hstack([X_text, X_numeric])

        print("\nFinal feature matrix:")
        print("Shape:", X.shape)

        return X
