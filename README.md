

#  Twitter Financial News Sentiment Classification

This repository contains an **end-to-end sentiment classification pipeline** for financial news tweets.
The project leverages the **[zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)** dataset from Hugging Face, trains and evaluates multiple machine learning models, and deploys the best-performing classifier for practical use.

---

##  Project Structure

```
📂 Text_Classification_on_zeroshot_twitter_financial_news_sentiment
│── Text_Classification_on_zeroshot_twitter_financial_news_sentiment.ipynb   # Model training & experimentation
│── Test_The_Deployed_Model_With_Sample_Text_Inputs.ipynb                     # Test deployed model on sample texts
│── sentiment_classifier.joblib                                              # Saved model & vectorizer (generated after training)
```

---

##  Workflow

### 1. Dataset Loading & Exploration

* Used **Hugging Face Datasets** library to load the dataset.

* Dataset contains **financial tweets labeled as sentiment classes**:

  * `0` → Negative
  * `1` → Neutral
  * `2` → Positive

* Example entries:

  ```
  Text: "$AAPL stock hits all-time high after strong earnings report."
  Label: 2 (Positive)
  ```

---

### 2. Data Preprocessing

* **Tokenization** with `CountVectorizer` and `TfidfVectorizer`.
* Splitting into **train, validation, and test sets**.
* Converted text data into sparse matrices or embeddings depending on model type.

---

### 3. Model Training & Evaluation

We experimented with **three approaches**:

#### 🔹 Multinomial Naive Bayes (CountVectorizer)

* Accuracy: \~78.8%
* F1 Score: \~78%
* **Best-performing baseline**.

#### 🔹 Multinomial Naive Bayes (TF-IDF Vectorization)

* Accuracy: \~72%
* F1 Score: \~63.5%
* Slightly worse than count-based features.

#### 🔹 Logistic Regression (Word2Vec Embeddings)

* Accuracy: \~68%
* F1 Score: \~58%
* Struggled with semantic understanding due to averaging embeddings.

 **Conclusion:**
`Multinomial Naive Bayes + CountVectorizer` gave the **best performance** and was chosen for deployment.

---

### 4. Model Deployment

* Saved trained model and vectorizer into a single file using `joblib`.

```python
import joblib
joblib.dump((classifier, vectorizer), 'sentiment_classifier.joblib')
```

* Reload model for inference:

```python
classifier, vectorizer = joblib.load('sentiment_classifier.joblib')
```

---

### 5. Model Testing with Sample Inputs

In `Test_The_Deployed_Model_With_Sample_Text_Inputs.ipynb`, we tested the deployed model:

```python
sample_text_1 = "$AAPL stock hits all-time high after strong earnings report."
sample_text_2 = "$TSLA announces plans to build a new Gigafactory in Texas."
sample_text_3 = "Investors are cautious as market volatility increases."
sample_text_4 = "$AMZN shares plunge following disappointing quarterly results."
```

**Predicted Results:**

| Sample Text                                                      | Predicted Sentiment                               |
| ---------------------------------------------------------------- | ------------------------------------------------- |
| `$AAPL stock hits all-time high after strong earnings report.`   | Neutral (1)                                       |
| `$TSLA announces plans to build a new Gigafactory in Texas.`     | Positive (2)                                      |
| `Investors are cautious as market volatility increases.`         | Positive (2) *(misclassified, expected Negative)* |
| `$AMZN shares plunge following disappointing quarterly results.` | Negative (0)                                      |

---

##  Insights

* Model performs **well on positive/negative financial news**.
* Occasionally misclassifies **cautious/uncertain language** as positive.
* Suggestion: Use **contextual embeddings (e.g., BERT/RoBERTa)** to better capture sentiment nuances.

---

##  Tech Stack

* **Python 3**
* **Pandas, NumPy**
* **Scikit-learn** (Naive Bayes, Logistic Regression)
* **Hugging Face Datasets**
* **Gensim (Word2Vec)**
* **Joblib (model persistence)**

---

##  How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Text_Classification_on_zeroshot_twitter_financial_news_sentiment.git
   cd Text_Classification_on_zeroshot_twitter_financial_news_sentiment
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn datasets gensim joblib
   ```

3. Open and run the notebooks in order:

   * `Text_Classification_on_zeroshot_twitter_financial_news_sentiment.ipynb` → Train & evaluate models.
   * `Test_The_Deployed_Model_With_Sample_Text_Inputs.ipynb` → Test deployed model predictions.

---

##  Future Improvements

* Implement **transformer-based models** (FinBERT, RoBERTa).
* Apply **hyperparameter tuning** (GridSearchCV, RandomizedSearchCV).
* Expand dataset with **real-time Twitter financial data**.
* Build an **API for real-time sentiment inference**.

---

##  Author

**Naima Aqeel**
📧 Email: `naimapnes@gmail.com`
🌐 GitHub: [NaimaAqeel](https://github.com/NaimaAqeel)

---

