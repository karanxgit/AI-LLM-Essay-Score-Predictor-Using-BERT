---

# **📝 AI-Powered Automated Essay Scoring & Explanation System**
This project leverages **Artificial Intelligence (AI) and Large Language Models (LLMs)** to automate **essay scoring and feedback generation**. Using a combination of **BERT embeddings, TF-IDF, a neural network-based regression model, and GPT-powered explanations**, this system predicts essay scores and provides dynamic feedback.

---

## 📌 **Table of Contents**
1. Introduction  
2. Installation  
3. Usage  
4. Preprocessing & Feature Engineering  
5. Model Training & Evaluation  
6. Score Prediction & Explanation  
7. Conclusion  

---

## 📌 **Introduction**
This AI-driven system utilizes state-of-the-art **Natural Language Processing (NLP) models** for:
✅ **Extracting essay features using DistilBERT embeddings & TF-IDF**  
✅ **Training a deep learning model for automated scoring**  
✅ **Generating textual explanations for predicted scores using GPT-2**  
✅ **Summarizing essays using BART for better interpretability**  

The project is designed to **mimic human evaluators**, making it ideal for **automated grading in education and research**.

---

## 📌 **Installation**
To set up the project, install the required Python dependencies:

```bash
pip install transformers torch datasets scikit-learn numpy pandas nltk torchmetrics
```

Additionally, download necessary **NLTK stopwords**:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 📌 **Usage**
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your_username/AI_Essay_Scorer.git
cd AI_Essay_Scorer
```

### 2️⃣ **Run the Essay Scorer**
```bash
python main.py --input_file "sample_essays/essay1.txt"
```

### 3️⃣ **Process an Image-Based Essay (OCR)**
```bash
python main.py --image "sample_essays/essay3.jpg"
```

---

## 📌 **Preprocessing & Feature Engineering**
The system combines **BERT embeddings** and **TF-IDF vectorization** for feature extraction.  
✔ **Text Cleaning:** Removes punctuation, stopwords, and converts text to lowercase.  
✔ **Tokenization:** Splits essays into meaningful words using NLTK.  
✔ **Feature Representation:**  
   - Uses **BERT embeddings** for a sample of training essays.  
   - Applies **TF-IDF vectorization** to all essays for efficiency.  
✔ **Hybrid Embedding Approach:** If a BERT embedding is unavailable, **TF-IDF is used as a fallback**.  

---

## 📌 **Model Training & Evaluation**
### **Training Process**
1️⃣ **Generate Embeddings:**  
   - Convert essays into **DistilBERT embeddings** or **TF-IDF vectors**.  
2️⃣ **Train a DistilBERT-based Regression Model:**  
   - Architecture: **3-layer neural network (768 → 256 → 128 → 1 output score)**  
   - Optimizer: **Adam (learning rate = 0.0003)**  
   - Loss function: **Mean Squared Error (MSE)**  
3️⃣ **Train on 3000 sampled essays** for computational efficiency.  

### **Evaluation Metrics**
✔ **Mean Absolute Error (MAE)**  
✔ **Confusion Matrix for Score Prediction**  
✔ **Training Loss Visualization**  

To retrain the model:
```bash
python train_model.py
```

---

## 📌 **Score Prediction & Explanation**
Once trained, the system predicts essay scores and **generates an explanation** using GPT-2 and BART.

### **How It Works:**
1️⃣ **BERT Embeddings** extract deep contextual information from the essay.  
2️⃣ **Neural Network Model** predicts a **score between 0-10**.  
3️⃣ **Summarization (BART)** condenses long essays into a **key summary**.  
4️⃣ **GPT-2 Explanation Generation** provides detailed **feedback on essay strengths & weaknesses**.  

### **Example Usage**
```python
essay_text = "Your essay content here"
predicted_score, explanation = score_text(essay_text)

print("Predicted Score:", predicted_score)
print("Generated Explanation:", explanation)
```

---

## 📌 **Conclusion**
This AI-powered essay scorer successfully predicts scores with high accuracy and generates human-like feedback.  
✅ **Combines BERT embeddings with TF-IDF for efficient text representation**  
✅ **Deep learning model improves automated scoring accuracy**  
✅ **GPT-2 dynamically generates justifications for assigned scores**  
✅ **BART-based summarization enhances readability of feedback**  

---

## 🚀 **Future Work**
🔹 **Fine-tuning BERT for domain-specific essay evaluation**  
🔹 **Exploring transformer models like T5 for scoring refinement**  
🔹 **Deploying as a web app using Flask or Streamlit**  

---

💡 **Want to contribute?**  
Feel free to fork the repository, submit pull requests, or suggest improvements!  
🌟 **Star this project if you found it useful!**  

🚀 **Happy Coding!** 🚀  

---
