!pip install transformers torch datasets scikit-learn numpy pandas

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from transformers import DistilBertTokenizer, DistilBertModel, pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# Load dataset
df = pd.read_csv("/content/Data_csv.csv")
df.drop("Unnamed: 0", inplace=True, axis=1)

# Split data into training and testing sets
y = df['final_score']
X = df[['essay']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert essays to lists
train_essays = X_train['essay'].tolist()
test_essays = X_test['essay'].tolist()

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = distilbert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token output

# Select 3000 random essays from training set for BERT embeddings
sample_size = 3000
random_indices = random.sample(range(len(train_essays)), sample_size)
sampled_essays = [train_essays[i] for i in random_indices]

sampled_train_vectors = np.array([get_bert_embedding(essay) for essay in sampled_essays])

!pip install torchmetrics

from torchmetrics.regression import MeanAbsoluteError
# Use TF-IDF for all essays
vectorizer = TfidfVectorizer(max_features=768)
tfidf_train_vectors = vectorizer.fit_transform(train_essays).toarray()
full_train_vectors = np.zeros((len(train_essays), 768))
for idx, essay in enumerate(train_essays):
    if idx in random_indices:
        bert_idx = list(random_indices).index(idx)
        full_train_vectors[idx] = sampled_train_vectors[bert_idx]
    else:
        full_train_vectors[idx, :] = tfidf_train_vectors[idx]

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

class EssayDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = EssayDataset(full_train_vectors, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class DistilBERTRegressor(nn.Module):
    def __init__(self):
        super(DistilBERTRegressor, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output single score

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBERTRegressor().to(device)
criterion = nn.MSELoss()

mae_metric = MeanAbsoluteError().to(device)
optimizer = Adam(model.parameters(), lr=0.0003)

# Train the model
epochs = 75
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    mae_score = mae_metric(predictions, labels)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, MAE: {mae_score:.4f}")


    # Save the best model
    torch.save(model.state_dict(), "/content/best_distilbert_model.pth")

print("Training complete! Best model saved.")


# Load pre-trained GPT model for explanation generation
generator = pipeline("text-generation", model="gpt2")

def score_text(text):
    essay_vector = get_bert_embedding(text)
    essay_vector = torch.tensor(essay_vector, dtype=torch.float32).to(device)
    essay_vector = essay_vector.view(1, -1)

    with torch.no_grad():
        score_prediction = model(essay_vector).cpu().item()

    # Ensure score is between 0-10
    score_prediction = max(0, min(score_prediction, 10))
    
    # Round to two decimal places
    rounded_score = round(score_prediction, 2)

    # Generate explanation dynamically
    prompt = f"The essay received a score of {rounded_score}. Explain the strengths and weaknesses."
    explanation = generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]

    return rounded_score, explanation

direct_text ="""Your essay goes here"""

if direct_text:
    predicted_score, explanation = score_text(direct_text)
    print("Predicted Score:", predicted_score)
    print("Explanation:", explanation)
else:
    print("Please provide valid text input.")
