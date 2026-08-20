# 😊 Emotion Detection

An NLP-based machine learning project that predicts the emotion
expressed in a piece of text.

The project uses **text preprocessing, TF-IDF vectorization, and machine
learning classification** to classify text into six emotion categories.
A Streamlit web application provides an interactive interface for making
predictions.

## 🚀 Live Demo

👉 [Try the live Emotion Detection
app](https://emotions-detection-1-82h8.onrender.com/)



## 📊 Dataset

The project uses the **Emotions Dataset for NLP** from Kaggle.

**Dataset:**
https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

The dataset provides three files:

-   `train.txt`
-   `test.txt`
-   `val.txt`

Each record contains a text sentence and its corresponding emotion
label.

Example:

``` text
i didnt feel humiliated ; sadness
i am feeling grouchy ; anger
```

## 🎯 Emotion Classes

The dataset contains six emotion categories:

    Label Emotion
  ------- ----------
        0 Sadness
        1 Anger
        2 Love
        3 Surprise
        4 Fear
        5 Joy

The deployed application maps the model's numerical predictions to these
emotion names.

## 🧠 Project Workflow

``` text
Raw Text
   ↓
Lowercase Conversion
   ↓
Remove Punctuation
   ↓
Remove Numbers
   ↓
Remove Non-ASCII Characters / Emojis
   ↓
Remove Stopwords
   ↓
TF-IDF Vectorization
   ↓
Machine Learning Classifier
   ↓
Emotion Prediction
   ↓
Streamlit Web App
```

## 🔧 Text Preprocessing

The training notebook applies the following preprocessing steps:

1.  Convert text to lowercase.
2.  Remove punctuation.
3.  Remove numeric characters.
4.  Remove non-ASCII characters.
5.  Remove English stopwords.

The same core cleaning process is used by the deployed Streamlit
application before generating predictions.

## 📐 TF-IDF Feature Engineering

The project uses `TfidfVectorizer` with:

``` python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

This means the model uses both:

-   Unigrams --- individual words
-   Bigrams --- pairs of consecutive words

`min_df=2` removes terms that occur only once, while `max_df=0.95`
removes extremely common terms. `sublinear_tf=True` applies sublinear
term-frequency scaling.

## 🤖 Model Experiments

Several classification algorithms were evaluated using the TF-IDF
representation.

  Model                                        Accuracy
  ---------------------------------------- ------------
  Multinomial Naive Bayes + Bag of Words         76.81%
  Multinomial Naive Bayes + TF-IDF               70.19%
  Logistic Regression + TF-IDF                   86.63%
  Linear SVM + TF-IDF                        **90.53%**
  SGD Classifier + TF-IDF                        90.31%
  Random Forest + TF-IDF                         88.75%
  KNN + TF-IDF                                   72.88%

### Best Experimental Model

The best result in the notebook was obtained with **Linear SVM**:

**Accuracy: 90.53%**

However, the model saved for the deployed application is **Logistic
Regression**, which achieved **86.63%** accuracy in the notebook.

This distinction is intentional in this README so the documented
deployment matches the actual saved model.

## 📈 Logistic Regression Performance

The Logistic Regression model used for the deployed application
achieved:

**Accuracy: 86.63%**

The notebook also evaluated Linear SVM in more detail:

  Class        Precision   Recall   F1-score
  ---------- ----------- -------- ----------
  Sadness           0.93     0.95       0.94
  Anger             0.91     0.88       0.90
  Love              0.85     0.80       0.82
  Surprise          0.86     0.72       0.78
  Fear              0.86     0.85       0.86
  Joy               0.91     0.94       0.93

Linear SVM overall:

-   Accuracy: **90.53%**
-   Macro F1-score: **0.87**
-   Weighted F1-score: **0.90**

## 🏆 SVM Hyperparameter Experiment

Different values of `C` were tested for Linear SVM:

      C     Accuracy
  ----- ------------
    0.5       90.44%
      1   **90.53%**
      2       90.38%
      5       89.69%
      7       89.66%
     11       89.34%
     19       88.56%

`C=1` produced the best accuracy among the tested values.

## 💾 Saved Model Files

The trained artifacts are saved using Joblib:

``` text
emotion_model.pkl
tfidf_vectorizer.pkl
```

The Streamlit application loads these files at runtime.

``` python
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
```

## 🌐 Streamlit Application

The frontend is implemented using Streamlit.

The application:

1.  Accepts text from the user.
2.  Cleans the text.
3.  Converts it into TF-IDF features.
4.  Passes the features to the trained model.
5.  Maps the numerical prediction to an emotion.
6.  Displays the detected emotion.

## 📁 Project Structure

``` text
Emotions-Detection/
│
├── main.py
├── emotion_model.pkl
├── tfidf_vectorizer.pkl
├── emotions-for-nlp.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Important Files

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `main.py`                           Streamlit application and
                                      prediction logic

  `emotion_model.pkl`                 Saved Logistic Regression model
                                      used by the app

  `tfidf_vectorizer.pkl`              Saved TF-IDF vectorizer

  `emotions-for-nlp.ipynb`            Data preprocessing, model
                                      experiments and evaluation

  `requirements.txt`                  Python dependencies
  -----------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the repository

``` bash
git clone https://github.com/Kunalthakur01/Emotions-Detection.git
cd Emotions-Detection
```

### 2. Create a virtual environment

``` bash
python -m venv venv
```

### 3. Activate the environment

**Windows**

``` bash
venv\Scripts\activate
```

**Linux/macOS**

``` bash
source venv/bin/activate
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

### 5. Run the application

``` bash
streamlit run main.py
```

The application will open in your browser.

## 📦 Dependencies

The application uses:

``` text
streamlit
numpy==1.26.4
scikit-learn==1.6.1
joblib==1.4.2
nltk==3.9.1
```

## 🧪 Example

Input:

``` text
I am extremely happy today!
```

Possible output:

``` text
Detected Emotion
Joy
```

The prediction is generated by the trained model and is not a rule-based
lookup.

## 🔍 Confusion Matrix

The notebook also evaluates the Linear SVM using a confusion matrix.

The main observed classification challenge is between emotionally
similar categories, particularly **Love/Joy** and some of the less
represented classes such as **Surprise**.

## 📚 What I Learned

This project helped me practice:

-   Natural Language Processing
-   Text preprocessing
-   Stopword removal
-   TF-IDF vectorization
-   N-gram feature extraction
-   Model comparison
-   Logistic Regression
-   Linear SVM
-   Hyperparameter tuning
-   Classification metrics
-   Confusion matrix analysis
-   Model serialization with Joblib
-   Streamlit development
-   ML model deployment

## 🔮 Future Improvements

-   Use the best-performing Linear SVM model in the deployed
    application.
-   Add prediction confidence / decision scores.
-   Display probabilities or confidence where supported by the model.
-   Experiment with word embeddings.
-   Experiment with deep learning approaches.
-   Try transformer-based models such as BERT.
-   Add batch prediction.
-   Improve UI/UX.
-   Add emotion distribution visualizations.
-   Evaluate the final deployed model on the provided validation/test
    sets separately.

## 👨‍💻 Author

**Kunal Singh**

Machine Learning & Data Science Enthusiast

-   GitHub: https://github.com/Kunalthakur01

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on
GitHub.
