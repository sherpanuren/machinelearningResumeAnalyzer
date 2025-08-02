# Resume Classifier Using Machine Learning

This project uses a machine learning classifier to categorize resumes into predefined job categories based on their content. The classifier is trained using various NLP techniques such as **TF-IDF** for feature extraction and a supervised learning algorithm (e.g., **SVM**, **Random Forest**, etc.) for classification.

## Project Overview

The goal of this project is to predict the job category of a given resume based on its content. This can be useful in resume sorting systems, where resumes need to be categorized for easier evaluation and decision-making.

### Key Features:
- **Text Preprocessing**: Cleans and preprocesses resumes (removes stopwords, tokenizes text, etc.).
- **TF-IDF Vectorization**: Converts the text data into numerical vectors that represent the importance of words in the resumes.
- **Prediction**: Uses a trained machine learning classifier to predict the job category based on the resume content.

## Setup

### Prerequisites
Make sure you have **Python** installed on your system. You'll also need to install some Python libraries. Use the following commands to install the required libraries:

```bash
pip install numpy pandas scikit-learn nltk pickle-mixin
