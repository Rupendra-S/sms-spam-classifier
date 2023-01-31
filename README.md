# sms-spam-classifier
# Introduction
This is a Natural Language Processing (NLP) based project for classifying SMS messages as either spam or not spam. The project uses machine learning algorithms to train a model on a labeled dataset of SMS messages and then use this trained model to predict the label of new SMS messages.The project is built using the Streamlit framework, allowing for an easy and intuitive user interface.

# Dependencies
The project requires the following libraries:

- Pandas
- Scikit-learn
- Numpy
- Matplotlib
- Seaborn
- NLTK
- Streamlit
# Dataset
The dataset used in this project is the SMS Spam Collection dataset, which can be downloaded from the kaggle.(https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

# Model Building
The following steps are involved in building the SMS Spam Classifier:

- Data preprocessing and cleaning
- Feature extraction using the Tf-Idf Vectorizer
- Model training and evaluation using BernoulliNB, MultinomialNB, and GaussianNB
- Model selection based on evaluation metrics (e.g. accuracy, precision)
- Model tuning to improve performance

# Results
After comparing the performance of different classifier algorithms, it was found that the **Multinomial Naive Bayes (MNB)** algorithm achieved the best precision results among the algorithms tested.
# Usage
To start the SMS Spam Classifier app, run the app.py script in your Python environment using the following command:
streamlit run app.py
This will launch the app in your web browser, allowing you to enter SMS messages and receive predictions on their spam/not spam status.

# Conclusion
This project demonstrates the effectiveness of NLP techniques in building a SMS Spam Classifier using machine learning algorithms. The trained MultinomialNB model achieved the best performance among the tested algorithms, with an precision of over **99%** and accuracy of **97%** on the test dataset, demonstrating its ability to accurately classify SMS messages as spam or not spam.
