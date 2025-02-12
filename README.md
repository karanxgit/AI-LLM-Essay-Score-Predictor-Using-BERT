---

# 📝 AI-Powered Automated Essay Scoring & Explanation System 

This project aims to **automate essay scoring** by leveraging **Artificial Intelligence (AI) and Large Language Models (LLM)**.  
By using **BERT embeddings, an LSTM model for scoring, and GPT-powered explanations**, the system can analyze essays, **predict scores**, and provide **constructive feedback**.  

---

## **📌 Table of Contents**
- [Introduction](#introduction)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Preprocessing](#preprocessing)  
- [Analysis](#analysis)  
- [Model Training](#model-training)  
- [Evaluation](#evaluation)  
- [Conclusion](#conclusion)  

---

## **📌 Introduction**
This project utilizes Python libraries such as **Transformers (BERT), TensorFlow/Keras (LSTM), NLTK, OpenCV (OCR), and GPT (Hugging Face)** for:  
✅ Extracting text from various sources (images, PDFs, DOCX, TXT)  
✅ Preprocessing essays for analysis  
✅ Generating **BERT embeddings** for feature extraction  
✅ Training an **LSTM model** for score prediction  
✅ Using **GPT (or LLaMA-2)** for generating explanations based on the score  

This AI-driven approach **mimics human evaluators** and enhances **automated grading systems** for education and research.

---

## **📌 Installation**
To run the project, install Python along with the required dependencies:  

```bash
pip install -r requirements.txt
```

Additionally, download the necessary NLTK stopwords and tokenizers:  
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## **📌 Usage**
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/karanxgit/AI_LLM_Essay_Scorer.git
cd AI_LLM_Essay_Scorer
```
  
2️⃣ **Run the essay scorer**  
```bash
python main.py --input_file "sample_inputs/essay1.txt"
```

3️⃣ **To process an image-based essay (OCR)**  
```bash
python main.py --image "sample_inputs/essay3.jpg"
```

---

## **📌 Preprocessing**
The preprocessing step includes:  
✔️ **Cleaning text** (removing punctuations, stopwords, and lowercasing)  
✔️ **Tokenizing sentences** using NLTK  
✔️ **Converting words into numerical features** using **BERT embeddings**  

This ensures that the text data is **structured and ready** for AI-based analysis.

---

## **📌 Analysis**
The analysis phase involves:  
- **Exploring essay scores & distributions**  
- **Visualizing data using Matplotlib & Seaborn**  
- **Converting text into vectors using BERT embeddings**  

This step helps **understand scoring patterns** and optimizes the grading process.

---

## **📌 Model Training**
The model training phase involves:  
- **Generating BERT embeddings** for each essay  
- **Splitting the dataset into training & testing sets**  
- **Training an LSTM model on the embedded text features**  
- **Saving the trained model for future use**  

To retrain the model, use:  
```bash
python scripts/train_model.py
```

---

## **📌 Evaluation**
The model is evaluated using:  
✔ **Accuracy score**  
✔ **Confusion matrix** (true/false positives & negatives)  
✔ **Performance visualization**  

---

## **📌 Conclusion**
This AI + LLM-based essay scorer successfully predicts essay scores with high accuracy.  
✅ **LSTM + BERT embeddings improve text understanding**  
✅ **GPT-generated feedback enhances automated evaluations**  
✅ **OCR integration expands input format flexibility**  

🚀 **Future Work:**  
🔹 Fine-tuning BERT for better feature extraction  
🔹 Testing additional AI models (e.g., Transformers, T5)  
🔹 Deploying the model via a **Flask/Streamlit web app**  


---

### **💡 Want to contribute?**  
- Feel free to **fork the repository**, submit **pull requests**, or suggest improvements!  
- Star ⭐ the project if you found it useful!  

🚀 Happy Coding! 🚀  

---
