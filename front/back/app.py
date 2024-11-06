from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


app = Flask(__name__)

# Load dataset from the provided file
data = pd.read_csv('text_emotion_prediction.csv', encoding='ISO-8859-1')

# Display the first few rows of the dataset
print(data.head())

# Data cleaning function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower()  # Convert to lowercase

# Apply preprocessing to comments
data['Comment'] = data['Comment'].apply(preprocess_text)

# Show how many records belong to each class
print("Records per class:")
print(data['Emotion'].value_counts())

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit and transform the text data
X_vectorized = vectorizer.fit_transform(data['Comment'])

# Define features and labels
X = X_vectorized
y = data['Emotion']  # Assuming 'Emotion' is the column name for labels

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate on test set
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model and vectorizer for future use
joblib.dump(model, 'text_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Test comment
comment_to_test = "I am so happy with my new job!"

# Preprocess the comment
cleaned_comment = preprocess_text(comment_to_test)

# Vectorize the cleaned comment
vectorized_comment = vectorizer.transform([cleaned_comment])

# Predict the emotion
predicted_emotion = model.predict(vectorized_comment)

# Output the result
print(f"Predicted Emotion: {predicted_emotion[0]}")

# Data cleaning function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower()  # Convert to lowercase

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    comment = data['comment']
    
    # Preprocess the text input
    comment_cleaned = preprocess_text(comment)
    
    # Vectorize and predict
    comment_vectorized = vectorizer.transform([comment_cleaned])
    
    prediction = model.predict(comment_vectorized)
    
    return jsonify({'emotion': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)