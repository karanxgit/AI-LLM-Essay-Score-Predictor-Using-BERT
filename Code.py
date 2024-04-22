!pip install nltk gensim keras sklearn
!pip install easyocr
!pip install opencv-python-headless
import numpy as np
import pandas as pd
import nltk
import re
import cv2
import numpy as np
import easyocr
import os
from keras.models import load_model
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential
import keras.backend as K
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors
import easyocr
nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv("/content/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1')
df.dropna(axis=1, inplace=True)
df.drop(columns=['domain1_score', 'rater1_domain1', 'rater2_domain1'], inplace=True, axis=1)

temp = pd.read_csv("/content/Processed_data - Processed_data.csv.csv")
temp.drop("Unnamed: 0", inplace=True, axis=1)
df['domain1_score'] = temp['final_score']

# Split data into features and target
y = df['domain1_score']
df.drop('domain1_score', inplace=True, axis=1)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert essays to lists
train_e = X_train['essay'].tolist()
test_e = X_test['essay'].tolist()

train_sents = []
test_sents = []

# Preprocess essays
stop_words = set(stopwords.words('english')) 

def sent2word(x):
    x = re.sub("[^A-Za-z]"," ",x)
    x.lower()
    filtered_sentence = [] 
    words = x.split()
    for w in words:
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words = []
    for i in raw:
        if(len(i) > 0):
            final_words.append(sent2word(i))
    return final_words

for i in train_e:
    train_sents += essay2word(i)

for i in test_e:
    test_sents += essay2word(i)

# Define and train the Word2Vec model
num_features = 300    
min_word_count = 40   
num_workers = 4       
context = 10          
downsampling = 1e-3   

model = Word2Vec(train_sents, 
                 workers=num_workers, 
                 vector_size=num_features, 
                 min_count=min_word_count, 
                 window=context, 
                 sample=downsampling)

model.init_sims(replace=True)  
model.wv.save_word2vec_format('/content/word2vecmodel.bin', binary=True)

# Define functions to convert essays to vectors
def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.wv.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model.wv[i])
    if noOfWords:
        vec = np.divide(vec, noOfWords)
    return vec

def getVecs(essays, model, num_features):
    c = 0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c += 1
    return essay_vecs

clean_train = [sent2word(i) for i in train_e]
training_vectors = getVecs(clean_train, model, num_features)

clean_test = [sent2word(i) for i in test_e]
testing_vectors = getVecs(clean_test, model, num_features)

training_vectors = np.array(training_vectors)
testing_vectors = np.array(testing_vectors)

# Reshape train and test vectors to 3 dimensions. (1 represents one timestep)
training_vectors = np.reshape(training_vectors, (training_vectors.shape[0], 1, training_vectors.shape[1]))
testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))

def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model
lstm_model = get_model()
lstm_model.fit(training_vectors, y_train, batch_size=64, epochs=150)
lstm_model.save('/content/final_lstm.h5')

# Load the Word2Vec model
w2v_model = KeyedVectors.load_word2vec_format('/content/word2vecmodel.bin', binary=True)

def extract_text_with_easyocr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(thresh, detail=0)
    extracted_text = " ".join(result)
    return extracted_text

def score_text(text, model, lstm_model, num_features):
    preprocessed_essay = sent2word(text)
    essay_vector = makeVec(preprocessed_essay, model, num_features)
    essay_vector = np.reshape(essay_vector, (1, 1, num_features))
    score_prediction = lstm_model.predict(essay_vector)
    
    # Ensure that score_prediction is a scalar value
    rounded_score = int(np.around(score_prediction[0][0]))  # Extract a single element from the array
    
    # Define the dictionary of essay explanations based on dataset's explanations
    processed_data = pd.read_csv("/content/Processed_data - Processed_data.csv.csv")
    essay_explanations = dict(zip(processed_data['final_score'], processed_data['explanation']))
    
    # Retrieve explanation based on the predicted score
    explanation = essay_explanations.get(rounded_score, "Explanation not found")
    
    return rounded_score, explanation
  
# Define functions to convert essays to vectors
def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)  # Change model.wv.index_to_key to model.index_to_key
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[i])  # Remove .wv
    if noOfWords:
        vec = np.divide(vec, noOfWords)
    return vec

# Define the dictionary of essay explanations
processed_data = pd.read_csv("/content/Processed_data - Processed_data.csv.csv")
essay_explanations = dict(zip(processed_data['final_score'], processed_data['explanation']))

image_path = "image path here"  # Set the correct path to your image
if os.path.exists(image_path):
    extracted_text = extract_text_with_easyocr(image_path)
    predicted_score, explanation = score_text(extracted_text, w2v_model, lstm_model, num_features)
else:
    new_essay = """essay goes here"""  # You should provide the actual essay text here
    predicted_score, explanation = score_text(new_essay, w2v_model, lstm_model, num_features)

print("Predicted Score:", predicted_score)
print("Explanation:", explanation)