from collections import Counter
from io import BytesIO
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)


class DataAnalyser:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.face_cascade = cv2.CascadeClassifier(
            "./haarcascade_frontalface_default.xml"
        )
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def analyze(self):
        print("Data Analysis:")
        print("Number of rows:", len(self.data))
        print("Number of columns:", len(self.data.columns))
        print("Columns:", self.data.columns.tolist())
        print("\nData Types:")
        print(self.data.dtypes)

        self.analyze_categories()
        self.analyze_structure()

        print("\nNumber of non-stopwords:")
        for (
            category,
            top_words,
        ) in self.analyze_number_of_non_stopwords().items():
            print(f"\nCategory: {category}")
            for word, count in top_words.items():
                print(f"{word}: {count}")

        # Bigrammok (n=2) és Trigrammok (n=3) elemzése
        self.analyze_ngrams(n=2, top_n=10)
        self.analyze_ngrams(n=3, top_n=10)

        # TF-IDF elemzés
        self.analyze_tfidf(top_n=10)

        self.analyze_title_and_description_length()
        self.analyze_images()

    def analyze_categories(self):
        print("Categories:")
        print(self.data["Category"].value_counts())

    def analyze_structure(self):
        print("\nData structure:")
        print(self.data.info())

    def analyze_number_of_non_stopwords(self, top_n: int = 10) -> dict:
        categories = self.data["Category"].unique()
        category_top_words = {}

        for category in categories:
            subset = self.data[self.data["Category"] == category]
            all_text = " ".join(
                (
                    subset["Title"].fillna("")
                    + " "
                    + subset["Description"].fillna("")
                ).tolist()
            )

            words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
            filtered_words = [w for w in words if w not in ENGLISH_STOP_WORDS]
            top_words = Counter(filtered_words).most_common(top_n)
            category_top_words[category] = dict(top_words)

        return category_top_words

    def analyze_title_and_description_length(self) -> None:
        for category in self.data["Category"].unique():
            subset = self.data[self.data["Category"] == category]
            title_lengths = subset["Title"].dropna().apply(len)
            description_lengths = subset["Description"].dropna().apply(len)

            print(f"\nCategory: {category}")
            print(
                f"Title Length: Mean = {title_lengths.mean():.2f}, "
                f"Std = {title_lengths.std():.2f}, "
                f"Median = {title_lengths.median():.0f}"
            )
            print(
                f"Description Length: Mean = {description_lengths.mean():.2f}, "
                f"Std = {description_lengths.std():.2f}, "
                f"Median = {description_lengths.median():.0f}"
            )

    def _calculate_colorfulness(self, image_rgb: np.ndarray) -> float:
        """Hasler és Süsstrunk metrika szerinti színesség."""
        R = image_rgb[:, :, 0].astype("float")
        G = image_rgb[:, :, 1].astype("float")
        B = image_rgb[:, :, 2].astype("float")

        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)

        std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
        mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))

        return float(std_root + (0.3 * mean_root))

    def _analyze_image(self, image_url: str) -> dict:
        default_res = {
            "brightness": 0.0,
            "colorfulness": 0.0,
            "face_count": 0,
            "person_count": 0,
        }
        try:
            resp = requests.get(image_url, timeout=10)
            if resp.status_code != 200:
                return default_res

            pil_image = Image.open(BytesIO(resp.content)).convert("RGB")
            image_rgb = np.array(pil_image)

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            brightness = float(np.mean(gray))
            colorfulness = self._calculate_colorfulness(image_rgb)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            )
            face_count = len(faces)

            boxes, _ = self.hog.detectMultiScale(
                image_bgr, winStride=(4, 4), padding=(8, 8), scale=1.05
            )
            person_count = len(boxes)

            return {
                "brightness": round(brightness, 2),
                "colorfulness": round(colorfulness, 2),
                "face_count": face_count,
                "person_count": person_count,
            }
        except Exception as e:
            print(f"Error occurred while analyzing image: {e}")
            return default_res

    def analyze_images(self) -> None:
        print("\nAnalyzing Images (Downloading and Feature Extraction)...")
        image_results = self.data["Image"].apply(self._analyze_image)

        image_features_df = pd.DataFrame(image_results.tolist())
        self.data = pd.concat([self.data, image_features_df], axis=1)

        print("\nImage Features Summary by Category:")
        summary = (
            self.data.groupby("Category")[
                ["brightness", "colorfulness", "face_count", "person_count"]
            ]
            .agg(["mean", "std", "median"])
            .round(2)
        )
        print(summary)

    def analyze_ngrams(self, n: int = 2, top_n: int = 10) -> None:
        for category in self.data["Category"].unique():
            category_subset = self.data[self.data["Category"] == category]
            texts = (
                category_subset["Title"].fillna("")
                + " "
                + category_subset["Description"].fillna("")
            )

            vectorizer = CountVectorizer(
                stop_words="english", ngram_range=(n, n)
            )
            X = vectorizer.fit_transform(texts)

            frequencies = np.asarray(X.sum(axis=0)).flatten()
            ngrams = vectorizer.get_feature_names_out()

            result = pd.DataFrame({"ngram": ngrams, "frequency": frequencies})
            result = result.sort_values("frequency", ascending=False)

            print(f"\nTop {n}-grams in {category}:")
            print(result.head(top_n).to_string(index=False))

    def analyze_tfidf(self, top_n: int = 10) -> None:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=2000
        )
        all_texts = (
            self.data["Title"].fillna("")
            + " "
            + self.data["Description"].fillna("")
        )
        X = vectorizer.fit_transform(all_texts)

        feature_names = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
        tfidf_df["Category"] = self.data["Category"].values

        for category in self.data["Category"].unique():
            category_data = tfidf_df[tfidf_df["Category"] == category]
            mean_tfidf = (
                category_data.drop(columns="Category")
                .mean()
                .sort_values(ascending=False)
            )

            print(f"\nTop TF-IDF features in {category}:")
            print(mean_tfidf.head(top_n).to_string())