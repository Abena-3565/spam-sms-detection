# Spam SMS Detection

## Project Overview
The **Spam SMS Detection** project aims to classify SMS messages as spam or not spam using machine learning algorithms. It utilizes natural language processing (NLP) techniques to preprocess text data, including tokenization, stopword removal, and the use of the TF-IDF vectorizer for feature extraction. The machine learning model used for classification is **Multinomial Naive Bayes**, and the dataset is highly imbalanced, requiring techniques like oversampling to handle the class imbalance.

## Dataset
The dataset used in this project contains a collection of SMS messages labeled as either "ham" (not spam) or "spam". This dataset is publicly available and is often used as a benchmark for text classification tasks.

- Total messages: 5574
- Spam messages: 747
- Ham messages: 4827

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK (Natural Language Toolkit)
- TF-IDF
- Naive Bayes Classifier
- SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced data.

## Steps Involved
1. **Data Preprocessing**: Clean the SMS data by removing stopwords, special characters, and performing stemming/lemmatization.
2. **Feature Extraction**: Convert text data into numerical form using TF-IDF vectorization.
3. **Handling Imbalanced Data**: Use techniques like SMOTE for oversampling the minority class (spam).
4. **Modeling**: Train a Multinomial Naive Bayes model to classify SMS messages.
5. **Evaluation**: Evaluate the model using accuracy, precision, recall, and F1-score.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Abena-3565/spam-sms-detection.git
## Contact Information
Email: abenezeralz659@mail.com
LinkedIn: Abenezer Alemayehu
Phone: 0935651441
GitHub: Abena-3565

