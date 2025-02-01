!pip install nltk gensim keras sklearn easyocr opencv-python-headless
!pip install transformers torch datasets

import numpy as np
import pandas as pd
import nltk
import re
import cv2
import easyocr
import os
import torch
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("/content/training_set.tsv", sep='\t', encoding='ISO-8859-1')
df.dropna(axis=1, inplace=True)
df.drop(columns=['domain1_score', 'rater1_domain1', 'rater2_domain1'], inplace=True)

# Load additional data containing scores
temp = pd.read_csv("/content/Data_csv.csv")
temp.drop("Unnamed: 0", inplace=True, axis=1)
df['domain1_score'] = temp['final_score']

# Split data into training and testing sets
y = df['domain1_score']
df.drop('domain1_score', inplace=True, axis=1)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert essays to lists
train_essays = X_train['essay'].tolist()
test_essays = X_test['essay'].tolist()

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Function to preprocess and tokenize text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub("[^A-Za-z]", " ", text).lower()
    words = text.split()
    return [w for w in words if w not in stop_words]

# Function to convert text into BERT embeddings
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Convert essays into BERT embeddings
train_vectors = np.array([get_bert_embedding(essay) for essay in train_essays])
test_vectors = np.array([get_bert_embedding(essay) for essay in test_essays])

# Reshape data for LSTM model
train_vectors = np.reshape(train_vectors, (train_vectors.shape[0], 1, train_vectors.shape[2]))
test_vectors = np.reshape(test_vectors, (test_vectors.shape[0], 1, test_vectors.shape[2]))

# Define LSTM model
def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=(1, 768), return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    return model

# Train the model
lstm_model = get_model()
lstm_model.fit(train_vectors, y_train, batch_size=64, epochs=50)  
lstm_model.save('/content/final_lstm.h5')

# Load pre-trained GPT model for explanation generation
generator = pipeline("text-generation", model="gpt2")

# Function to predict score and generate explanation
def score_text(text):
    essay_vector = get_bert_embedding(text)
    essay_vector = np.reshape(essay_vector, (1, 1, 768))
    
    score_prediction = lstm_model.predict(essay_vector)
    rounded_score = int(np.around(score_prediction[0][0]))

    # Generate explanation dynamically
    prompt = f"The essay received a score of {rounded_score}. Explain the strengths and weaknesses."
    explanation = generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]

    return rounded_score, explanation

# Function to extract text from image (OCR)
def extract_text_with_easyocr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(gray, detail=0)
    return " ".join(result)

# Functions to extract text from different file formats
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Function to process different input types
def process_input(file_path=None, direct_text=None):
    extracted_text = ""

    if direct_text:
        extracted_text = direct_text
    elif file_path and os.path.exists(file_path):
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            extracted_text = extract_text_with_easyocr(file_path)
        elif file_path.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            extracted_text = extract_text_from_docx(file_path)
        elif file_path.lower().endswith('.txt'):
            extracted_text = extract_text_from_txt(file_path)
        else:
            print("Unsupported file type.")
    return extracted_text

# Example usage
file_path = "/content/sample_essays_to_test.txt"  # Replace with actual file path
direct_text = "Insert essay text here."  # Or replace with actual essay text

if direct_text or file_path:
    extracted_text = process_input(file_path=file_path, direct_text=direct_text)

    if extracted_text:
        predicted_score, explanation = score_text(extracted_text)
        print("Predicted Score:", predicted_score)
        print("Explanation:", explanation)
    else:
        print("No text extracted or provided.")
else:
    print("Please provide either a valid file path or direct text input.")
