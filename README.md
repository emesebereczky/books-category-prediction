# Data Science Project

## Authors

| Name | Student ID |
| :--- | :--- |
| **Attila Forgacs** | 2602175 |
| **Emese Bereczky** | 2602173 |

**Source Code:** [https://github.com/emesebereczky/books-category-prediction](https://github.com/emesebereczky/books-category-prediction)

## Installation & Setup
To install all required dependencies, clone the repository and run:

```bash
pip install -r requirements.txt
```

---

## Introduction

The goal of the project was to create a binary classifier that predicts if a certain book is fiction or nonfiction. We created a Python project to scrape, clean, and analyse data, and we trained a model from these data to make predictions. The project description provided the [books.toscrape.com](https://books.toscrape.com/) website to train and test our model. On this website, we can find the title, category, image, description, and similar information about books. We decided to use supervised learning, specifically logistic regression, for this project because the correct output (fiction or nonfiction) is known for each book.

---

## Data Scraping

For scraping, we created an `IWebScraper` interface in case we would like to switch between web scraper modules in the future. We decided to use the `BeautifulSoup` and `requests` modules.

From the `books.toscrape.com` website, we scraped the following information for each book:
- Title
- Category
- Description
- Image URL

We initialized two scraper instances: one for fiction and one for nonfiction. First, the instance scrapes the number of pages, then loops through the pages and scrapes the necessary information for each book using `BeautifulSoup`. The scraped data is stored in lists and ultimately returned as a `pandas` DataFrame.

---

## Data Cleaning and Analysis

In our data cleaner class, we remove any corrupted data such as missing values, duplicates, invalid image URLs, and non-English descriptions. We also clean text-based information by replacing whitespace variants with standard space characters.

```text
                                                   Title  ...                                        Description
0                                             Soumission  ...  Dans une France assez proche de la nÃ´tre, un ...
1                                 Private Paris (Private #10)  ...  Paris is burning--and only Private's Jack Morg...
2                                   We Love You, Charlie Freeman  ...  The Freeman family--Charles, Laurel, and their...
3                                                         Thirst  ...  On a searing summer Friday, Eddie Chapman has ...
4     The Murder That Never Was (Forensic Instincts #5)  ...  Given the opportunity, would you assume someon...
..                                                           ...  ...                                                ...
170                                                H is for Hawk  ...  When Helen Macdonald's father died suddenly on...
171               Travels with Charley: In Search of America  ...  An intimate journey across and in search of Am...
172                                                    The Tumor  ...  John Grisham says THE TUMOR is the most import...
173              The End of the Jesus Era (An Investigation #1)  ...  God exists!? This may seem unprovable. The tru...
174                                             Eat, Pray, Love  ...  In her early thirties, Elizabeth Gilbert had e...

[175 rows x 4 columns]
Initial number of rows: 175
Missing values:
Title          0
Category       0
Description    0
dtype: int64
Rows removed due to missing values: 0
Duplicate titles: 0
Duplicate rows removed: 0
Text formatting cleaned.
english: 174, non-english: 1
Non-English rows removed: 1
Valid image URLs: 174, Invalid image URLs: 0
No invalid image URLs found.
Final number of rows: 174
Data Analysis:
Number of rows: 174
Number of columns: 4
Columns: ['Title', 'Image', 'Category', 'Description']
```

We analysed the data across several factors. We checked the ratio of fiction to nonfiction books, the most frequent non-stopwords, 2-grams, 3-grams, TF-IDF features, title length, description length, and different properties of the image for each category.

The most outstanding result of the analysis was the title character length:
- **Fiction average:** $21.75$ characters
- **Nonfiction average:** $54.43$ characters

We also examined four properties of the book cover images:
- Colourfulness
- Brightness
- Count of faces
- Count of people

We used the `io.BytesIO`, `OpenCV`, and `Pillow` modules for image processing. Brightness and colourfulness showed minor differences, while face and person counts were near zero for both categories. This indicates that person/face detection was inaccurate on smaller book cover graphics.

---

## Feature Engineering

We created a feature engineering class that produces a feature matrix based on the exploratory analysis. Our final feature matrix contains **209 columns**, including:
- $200$ TF-IDF features
- Title length & title word count
- Description length & description word count
- Title length / description length ratio
- 4 image properties (brightness, colourfulness, person count, face count)

This class provides both `fit_transform` and `transform` methods:
- `fit_transform` is applied to the training data to learn parameters and vocabulary.
- `transform` is applied to the test data to convert it using the exact rules, vocabulary, and scaling learned from the training set without learning anything new.

This structure prevents data leakage and ensures an identical feature representation across datasets.

---

## Model Training

We applied supervised learning using the logistic regression model from `scikit-learn`'s linear models. 

1. The dataset is split into training and test sets.
2. The category labels are aligned accordingly.
3. After fitting and transforming the training features, the logistic regression model is trained.
4. The trained model predicts categories for the test set.

---

## Model Evaluation

The model correctly predicted **30 out of 35** categories, resulting in an overall accuracy of **86%**.

### Confusion Matrix

| | Predicted Fiction | Predicted Nonfiction |
| :--- | :---: | :---: |
| **Actual Fiction** | 12 | 1 |
| **Actual Nonfiction** | 4 | 18 |

- **False Positives:** 1
- **False Negatives:** 4
- **Fiction:** Precision = $75\%$, Recall = $92\%$
- **Nonfiction:** Precision = $95$, Recall = $82\%$

### $F_1$-Scores

$$\text{Fiction } F_1 = 2 \times \frac{0.92 \times 0.75}{0.92 + 0.75} \approx 0.83$$

$$\text{Nonfiction } F_1 = 2 \times \frac{0.82 \times 0.95}{0.82 + 0.95} \approx 0.88$$

---

### Evaluation Output

```text
Final feature matrix:
Shape: (35, 209)
X_train: (139, 209)
X_test: (35, 209)
              precision    recall  f1-score   support

     Fiction       0.75      0.92      0.83        13
  Nonfiction       0.95      0.82      0.88        22

    accuracy                           0.86        35
   macro avg       0.85      0.87      0.85        35
weighted avg       0.87      0.86      0.86        35


Wrong predictions:
                                                          Title    Category  Predicted
                                              Into the Wild  Nonfiction    Fiction
The Suffragettes (Little Black Classics, #96)  Nonfiction    Fiction
                                Letter to a Christian Nation  Nonfiction    Fiction
                                      Hamilton: The Revolution  Nonfiction    Fiction
             Finders Keepers (Bill Hodges Trilogy #2)     Fiction  Nonfiction
======================================================================
CONFUSION MATRIX
======================================================================
[[12  1]
 [ 4 18]]
```

---

## Conclusion and Discussion

In this project, we developed a machine learning model that predicts whether a book is fiction or nonfiction. Achieving an accuracy of $86\%$ on the test set ($30/35$ books correct) demonstrates solid classification performance.

The most influential features were textual. In particular, title character length exhibited noticeable differences across categories.

The non-textual inputs consisted of book cover images. While colourfulness and brightness were successfully integrated into the feature space, human presence metrics (face and person counts) were largely ineffective due to detector limitations on stylized, low-resolution book covers.

---

## References

- Books to Scrape: [https://books.toscrape.com/](https://books.toscrape.com/)