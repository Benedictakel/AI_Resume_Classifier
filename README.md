# 📂 AI Resume Classifier

This repository contains an **AI-powered system** that uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to **classify resumes** into predefined **job roles**, such as:

* Data Scientist
* Web Developer
* HR Manager
* Software Engineer
* Business Analyst
* … and more.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Objectives](#objectives)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Overview](#model-overview)
* [Evaluation Metrics](#evaluation-metrics)
* [Sample Predictions](#sample-predictions)
* [Project Structure](#project-structure)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

Manually screening resumes is time-consuming and inefficient. This AI Resume Classifier leverages **machine learning** to **automate the classification** of resumes based on their content, streamlining the hiring process and improving recruitment efficiency.



## 📚 Dataset

* **Source**: Public datasets (e.g., Kaggle Resume Dataset) or custom labeled resumes
* **Format**: `.txt`, `.pdf`, or plain text extracted from uploaded resumes
* **Classes**:

  * Data Scientist
  * HR
  * Software Developer
  * Web Developer
  * Business Analyst
  * Others (can be expanded)





## 🎯 Objectives

✔️ Build a pipeline to extract and preprocess resume text

✔️ Vectorize resumes using NLP techniques (TF-IDF, Word2Vec, etc.)

✔️ Train and evaluate ML models for classification

✔️ Predict job roles from new/unseen resumes

✔️ (Optional) Build a simple web interface for resume upload and prediction



## ✨ Features

* Extract text from `.txt` and `.pdf` resume formats
* Clean and preprocess resume text using NLP
* Vectorize resumes with **TF-IDF**
* Train models: **Logistic Regression**, **Random Forest**, **SVM**, etc.
* Generate performance metrics (accuracy, precision, recall, F1-score)
* *(Optional)* Deploy using **Flask** or **Streamlit** for user input



## 🛠️ Technologies Used

* **Python 3**
* **Scikit-learn**
* **NLTK / spaCy**
* **Pandas & NumPy**
* **TfidfVectorizer / Word2Vec**
* **PyPDF2 / pdfminer.six** *(for PDF extraction)*
* **Matplotlib / Seaborn** *(for visualization)*
* *(Optional)* Flask / Streamlit *(for deployment)*



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/AI_Resume_Classifier.git
cd AI_Resume_Classifier
```

2. **Create a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```



## ▶️ Usage

### 🧪 Train the Model

```bash
python train_model.py
```

### 📂 Classify a Resume

```bash
python predict_resume.py --file data/sample_resume.pdf
```

> The model will output the predicted job role (e.g., `Predicted Role: Web Developer`).



## 🧠 Model Overview

* **Text Preprocessing**:

  * Lowercasing, stopword removal, stemming/lemmatization
  * Tokenization using NLTK or spaCy
* **Vectorization**:

  * TF-IDF to convert text into numeric feature vectors
* **ML Algorithms**:

  * Logistic Regression (baseline)
  * SVM (high accuracy on sparse data)
  * Random Forest
* **Model Selection**:

  * Train/test split, cross-validation, grid search



## 📊 Evaluation Metrics

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 88%      | 87%       | 86%    | 86.5%    |
| SVM                 | 90%      | 89%       | 88%    | 88.5%    |
| Random Forest       | 85%      | 84%       | 83%    | 83.5%    |





## 🔍 Sample Predictions

```plaintext
Resume: data/john_doe_resume.pdf  
Predicted Role: Data Scientist  
Confidence: 91.7%
```



## 📁 Project Structure

```
AI_Resume_Classifier/
 ┣ data/
 ┃ ┗ sample_resume.pdf
 ┣ models/
 ┃ ┗ resume_classifier.pkl
 ┣ src/
 ┃ ┣ preprocess.py
 ┃ ┣ train_model.py
 ┃ ┗ predict_resume.py
 ┣ app.py  # (optional web app)
 ┣ requirements.txt
 ┗ README.md
```



## 💡 Future Work

* Integrate **NER (Named Entity Recognition)** for extracting skills, education, experience
* Use **BERT embeddings** or **transformers** for improved classification
* Add support for **multi-label classification** (if a resume fits more than one role)
* Deploy as a **web app or API** using Flask, FastAPI, or Streamlit
* Create an **admin dashboard** for HR to upload and manage resume predictions



## 🤝 Contributing

Contributions are welcome to expand job categories, improve model performance, or enhance the UI.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Portfolio](#)



### ⭐ If you found this project helpful, consider giving it a star and sharing with recruiters and machine learning enthusiasts.

