
# Resume Analyzer using NLP

This project uses a machine learning classifier to categorize resumes into predefined job categories based on their content. The classifier is trained using various NLP techniques such as **TF-IDF** for feature extraction and a supervised learning algorithm for classification.

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
````

### Files in the Repository

* **clf.pkl**: The trained machine learning classifier model.
* **resume\_classifier.py**: The Python script that loads the model, preprocesses the resume, and makes predictions.
* **myresume.txt**: A sample resume file (replace with your actual resume text).
* **cleanResume()**: A function used to preprocess the input resume text before prediction.

## How to Use

### Step 1: Load the Classifier

The first step is to load the trained classifier that was saved using **pickle**.

```python
import pickle

# Load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))
```

### Step 2: Clean the Input Resume

Next, you need to clean the resume text before making predictions. The cleaning process might involve:

* **Tokenization**
* **Removal of stopwords**
* **Lowercasing and stemming**

```python
# Clean the input resume
cleaned_resume = cleanResume(myresume)
```

Ensure that the function `cleanResume()` is implemented to process the resume properly (e.g., removing unwanted characters, stopwords, etc.).

### Step 3: Transform the Cleaned Resume

The cleaned resume text is then transformed into a feature vector using the **TF-IDF vectorizer** that was used to train the model.

```python
# Transform the cleaned resume using the trained TfidfVectorizer
input_features = tfidf.transform([cleaned_resume])
```

Ensure that the `tfidf` object used in training the model is loaded or available in the environment.

### Step 4: Make the Prediction

The classifier then predicts the job category based on the input resume features. The prediction is a category ID, which can be mapped to the corresponding job category name.

```python
# Make the prediction using the loaded classifier
prediction_id = clf.predict(input_features)[0]

# Map category ID to category name
category_mapping = {
    0: "Advocate",
    30: "TEACHER",
    22: "Java Developer",
    18: "HR",
    31: "Testing",
    14: "DevOps Engineer",
    11: "DIGITAL-MEDIA",
    27: "Python Developer",
    32: "Web Designing",
    19: "Hadoop",
    8: "CONSULTANT",
    29: "Sales",
    6: "Blockchain",
    16: "ETL Developer",
    12: "Data Science",
    25: "Operations Manager",
    23: "Mechanical Engineer",
    1: "AUTOMOBILE",
    3: "Arts",
    13: "Database",
    10: "DESIGNER",
    21: "INFORMATION-TECHNOLOGY",
    5: "BUSINESS-DEVELOPMENT",
    20: "Health and fitness",
    17: "Electrical Engineering",
    26: "PMO",
    15: "DotNet Developer",
    7: "Business Analyst",
    4: "Automation Testing",
    24: "Network Security Engineer",
    28: "SAP Developer",
    9: "Civil Engineer",
    2: "Advocate"
}

category_name = category_mapping.get(prediction_id, "Unknown")

print("Predicted Category:", category_name)
print(prediction_id)
```

### Sample Output:

If the resume text matches the job description of a **Data Scientist**, the prediction output would look like this:

```bash
Predicted Category: Data Science
Prediction ID: 12
```

### Step 5: Customize for Your Own Resumes

To classify your own resume, replace `myresume` with the path to your resume file or directly provide the text to the function.

```python
myresume = "path_to_resume.txt"
```

### Example Resume Classifier Script:

Here’s an example script (`resume_classifier.py`) that ties everything together:

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the classifier and TfidfVectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Assuming you saved the TfidfVectorizer

# Load and clean the resume
myresume = "path_to_resume.txt"
cleaned_resume = cleanResume(myresume)

# Transform the resume using the TfidfVectorizer
input_features = tfidf.transform([cleaned_resume])

# Make the prediction
prediction_id = clf.predict(input_features)[0]

# Mapping category ID to category name
category_mapping = {
    0: "Advocate",
    30: "TEACHER",
    22: "Java Developer",
    18: "HR",
    31: "Testing",
    14: "DevOps Engineer",
    11: "DIGITAL-MEDIA",
    27: "Python Developer",
    32: "Web Designing",
    19: "Hadoop",
    8: "CONSULTANT",
    29: "Sales",
    6: "Blockchain",
    16: "ETL Developer",
    12: "Data Science",
    25: "Operations Manager",
    23: "Mechanical Engineer",
    1: "AUTOMOBILE",
    3: "Arts",
    13: "Database",
    10: "DESIGNER",
    21: "INFORMATION-TECHNOLOGY",
    5: "BUSINESS-DEVELOPMENT",
    20: "Health and fitness",
    17: "Electrical Engineering",
    26: "PMO",
    15: "DotNet Developer",
    7: "Business Analyst",
    4: "Automation Testing",
    24: "Network Security Engineer",
    28: "SAP Developer",
    9: "Civil Engineer",
    2: "Advocate"
}

category_name = category_mapping.get(prediction_id, "Unknown")

# Output the predicted category
print(f"Predicted Category: {category_name}")
```

## Conclusion

* This project classifies resumes into predefined job categories using a machine learning classifier.
* The classifier is trained using **TF-IDF** and a supervised model, and predictions are based on the input resume.
* Future improvements could include using more sophisticated models like **BERT** or **GPT** for semantic analysis.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

* **Scikit-learn** for machine learning tools like the classifier and TF-IDF vectorizer.
* **Pickle** for saving and loading models.
* **NLTK/Spacy** for text preprocessing and tokenization.

````

---

### Steps to Follow:
1. Copy and paste the above content into your **`README.md`** file.
2. Save and commit the changes to GitHub using the following commands:

```bash
git add README.md
git commit -m "Added usage instructions and example script"
git push origin main
````
