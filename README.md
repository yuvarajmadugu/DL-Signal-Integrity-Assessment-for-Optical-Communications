# 📡 Hybrid AI Framework for High-Precision Signal Quality Classification in Optical and Wireless Communication Systems

## 📌 Overview
This project presents a **hybrid machine learning and deep learning framework** for robust signal quality classification in optical and wireless communication systems.

Modern communication networks face challenges such as **noise, dispersion, non-linearities, and hardware imperfections**, which degrade signal integrity. Traditional rule-based and single-model approaches fail to scale under these complex conditions.

To address this, the project introduces a **data-driven, automated pipeline** capable of learning complex patterns from signal data and accurately predicting signal quality and anomalies.

---

## 🚧 Problem Statement
Traditional signal assessment techniques:
- Rely on static thresholds or single-model learning  
- Perform poorly in noisy and imbalanced datasets  
- Fail to generalize across varying channel conditions  

These limitations make reliable signal integrity assessment difficult in real-world optical communication systems.

---

## 💡 Proposed Solution
We propose a **Hybrid AI Framework** combining Machine Learning, Deep Learning, and an ensemble-based fusion approach.

### 🔗 Dual-Learner Fusion Architecture (DLFA)
A key contribution of this project is the **DLFA**, which combines:

- **Logistic Regression (LR)** → probabilistic linear modeling  
- **Random Forest (RF)** → captures non-linear feature interactions  

These models are fused using a **soft voting (probability-based) ensemble**, resulting in:
- Improved prediction accuracy  
- Reduced overfitting  
- Better generalization under noisy conditions  

---

## 🎯 Objectives
- Analyze signal integrity issues in optical communication systems  
- Perform advanced preprocessing and feature engineering  
- Build and evaluate ML and DL models  
- Detect anomalies and predict signal degradation  
- Compare classical ML models with deep learning approaches  

---

## 🧠 Technologies & Tools

**Programming Language:**  
- Python  

**Libraries:**  
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- Seaborn  
- Imbalanced-learn (SMOTE)  
- Joblib  

**GUI:**  
- Tkinter  

**Environment:**  
- Jupyter Notebook / VS Code  

---

## 📊 Dataset Description
The dataset includes signal parameters from optical communication systems:

- Tx
- Rx
- Signal-to-Noise Ratio (SNR)  
- Bit Error Rate (BER)  
- Power Levels  
- Dispersion & noise-related metrics  
- Signal quality / anomaly labels  

> Dataset may be simulated or experimentally generated. Generally we used this dataset from kaggle website

---

## ⚙️ System Pipeline

### 1️⃣ Data Preprocessing
- Missing value handling  
- Label encoding  
- Feature scaling  
- SMOTE for class imbalance  

### 2️⃣ Model Training & Benchmarking
- Complement Naive Bayes (CNB)  
- Support Vector Machine (SVM)  
- Linear Discriminant Analysis (LDA)  
- Random Forest  
- Logistic Regression
- Proposed DLFA Model  

### 3️⃣ Evaluation Metrics
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC Curve & AUC  

### 4️⃣ Deployment Interface
A **Tkinter-based GUI** enabling:
- Dataset upload  
- Exploratory Data Analysis (EDA)  
- Model training & evaluation  
- Real-time signal quality prediction  

---

## 🧪 Models Implemented
- Complement Naive Bayes  
- Support Vector Machine (SVM)  
- Linear Discriminant Analysis (LDA)  
- Random Forest  
- Logistic Regression 
- **Dual-Learner Fusion Architecture (DLFA)**  

---

## 📈 Results & Performance
- DLFA achieved **higher accuracy and robustness** compared to individual models  
- Deep learning models performed well under **non-linear and noisy conditions**  
- Improved:
  - Generalization across diverse signal conditions  
  - Stability in imbalanced datasets  
  - Reliability in anomaly detection  

---

## 🖥️ How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yuvarajmadugu/DL-Signal-Integrity-Assessment-for-Optical-Communications.git
cd DL-Signal-Integrity-Assessment-for-Optical-Communications
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run GUI Application
```bash
python main.py
```
### 4️⃣ Run via Jupyter Notebook
```bash
jupyter notebook
```
---

## 📁 Project Structure
```bash
├── images/
│   └── images showing the project flow
├── models/
│   └── trained_models.joblib
├── results/
│   └── evaluation_outputs/
├── bg_img/
├── main.py
├── opticom signal quality dataset.csv
├── signal quality.ipynb
├── testdata.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Future Enhancements
Integration with real-time optical network monitoring systems
Advanced architectures (CNN-LSTM, Transformers)
Real-time visualization dashboard
Deployment as a web-based application

---

## 🤝 Contributors
Team Members  
Randhi Ram kiran  
Madugu Yuvaraj  
M.Rakesh

---

## 📜 License
This project is intended for academic and research purposes and been published in-  

---

## 🔗 Paper Publication
[Preview](https://zesterapublications.com/journals/index.php/ijaene/article/view/428)

---

## ⭐ Support
If you find this project useful, consider giving it a ⭐ on GitHub!








