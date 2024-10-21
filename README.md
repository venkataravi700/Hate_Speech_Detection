Overview
Hate speech on social media is a growing concern, and this project aims to create a machine learning model capable of classifying tweets into different categories of offensive or harmful speech. Using natural language processing (NLP) techniques, we process a dataset of labeled tweets to train and evaluate the model's performance.

Dataset
The dataset used in this project is a CSV file (labeled_data.csv) containing over 24,000 tweets, each labeled as:

Hate Speech (0)
Offensive Language (1)
Neither (2)
Each tweet also includes additional features like the number of occurrences of hate speech, offensive language, and the tweet text itself.

Installation
To run this project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/venkatatavi700/hate_speech_detection.git
Install the required packages:

bash
Copy code
pip install numpy pandas scikit-learn
Ensure you have the dataset labeled_data.csv in the project directory.

Usage
Load the dataset and explore the data:

python
Copy code
import pandas as pd
dataset = pd.read_csv('labeled_data.csv')
Preprocess the data, including tokenization and feature extraction, to prepare it for model training.

Train the model using classification algorithms like Logistic Regression or Random Forest, which are implemented in the Jupyter notebook.

Evaluate the model's performance using accuracy, precision, and recall metrics.

Model
The project applies a variety of machine learning algorithms to classify tweets, with the primary model being trained on labeled data. The pipeline includes:

Text preprocessing (tokenization, stopword removal, etc.)
Feature extraction using TF-IDF
Model training with classifiers like Logistic Regression, Naive Bayes, or SVM
Results
The model is evaluated based on its ability to correctly classify hate speech and offensive language, achieving a promising accuracy and F1-score on the test dataset. Further improvements can be made by fine-tuning the model and using advanced NLP techniques.
