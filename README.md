# 💳 UPI Fraud Detection System

🚀 A Machine Learning-based project to detect fraudulent UPI transactions using data analysis and predictive modeling.

---

## 📌 Overview

With the rapid growth of digital payments in India, UPI transactions have increased significantly. However, this has also led to a rise in fraud cases. This project focuses on building a system that can identify suspicious transactions using Machine Learning techniques.

The model analyzes transaction patterns and predicts whether a transaction is **fraudulent or legitimate**.

---

## 🎯 Objectives

- Detect fraudulent UPI transactions
- Analyze transaction patterns and anomalies
- Build a predictive ML model
- Improve financial security using data-driven insights

---

## 🧠 Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📂 Project Structure
```bash
UPI_Fraud_Detection/
├── data/
├── notebooks/
├── model/
├── app.py
├── requirements.txt
└── README.md
```



---

## ⚙️ How It Works

1. Data Collection (transaction dataset)
2. Data Preprocessing
   - Handling missing values
   - Encoding categorical data
   - Feature scaling
3. Model Training
   - Logistic Regression
   - Random Forest / Other ML models
4. Evaluation
   - Accuracy
   - Precision & Recall
   - Confusion Matrix
5. Prediction
   - Classifies transaction as **Fraud / Not Fraud**

---

## 📊 Features Used

- Transaction Amount
- Transaction Type
- Sender & Receiver details
- Time & Frequency
- Account balance behavior

These features help identify abnormal transaction patterns.

---

## 📈 Model Performance

- **Accuracy:** `99.97%` 🎯  
- High precision in detecting fraudulent transactions  
- Strong performance even with imbalanced data  
- Minimal false positives and false negatives  

> ⚠️ Note: Extremely high accuracy (like 99.97%) may indicate class imbalance in the dataset.  
> Hence, additional metrics such as **Precision, Recall, and F1-score** are also important for proper evaluation.

---

## ▶️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/AnshulKumar001/-UPI_Fraud_Detection.git

# Navigate to project
cd -UPI_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Run the project
python app.py
