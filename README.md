# IMDb Movie Review Sentiment Analysis

A machine-learning project that classifies IMDb movie reviews as **positive** or **negative**. The notebook covers data preparation, TF-IDF feature extraction, model training, evaluation, and error analysis.

## Objectives

* Clean and prepare movie-review text
* Convert text into TF-IDF features
* Train and compare classification models
* Evaluate predictions using accuracy, precision, recall, and F1 score
* Analyse false-positive and false-negative predictions
* Save the best model and vectorizer for reuse

## Dataset

The project uses the **IMDb Dataset of 50K Movie Reviews**.

| Column      | Description                     |
| ----------- | ------------------------------- |
| `review`    | Text of a movie review          |
| `sentiment` | Label: `positive` or `negative` |

Download the dataset as `IMDB Dataset.csv` and place it in the notebook’s working directory.

## Models Evaluated

* Logistic Regression
* Linear Support Vector Classifier (`LinearSVC`)
* Random Forest Classifier

## Results

| Model               |   Accuracy |  Precision | Recall |   F1 Score |
| ------------------- | ---------: | ---------: | -----: | ---------: |
| Logistic Regression |     90.88% |     90.04% | 92.08% |          - |
| LinearSVC           | **90.93%** | **90.56%** | 91.55% | **91.05%** |
| Random Forest       |     86.97% |     87.77% | 86.15% |     86.95% |

**LinearSVC achieved the best overall result** on the test data.

## Error Analysis

The notebook identifies misclassified reviews and separates them into:

* **False positives:** negative reviews predicted as positive
* **False negatives:** positive reviews predicted as negative

This helps examine difficult reviews containing mixed opinions, sarcasm, or ambiguous wording.

## Tech Stack

* Python
* Jupyter Notebook / Google Colab
* Pandas and NumPy
* Matplotlib and Seaborn
* scikit-learn
* Joblib

## Run Locally

```bash
git clone https://github.com/yupitsmegd7/ELABS_Probation.git
cd ELABS_Probation
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook
```

Open `Movie_Sentiment.ipynb`, add `IMDB Dataset.csv` to the project folder, and run all cells in order.

## Saved Files

After training, the notebook saves:

* `model.pkl` — trained LinearSVC
