# 🛠️ Predictive Maintenance of Milling Machines  
### AI-Driven Failure Prediction System

## 📋 Project Overview
This repository contains a **complete end-to-end predictive maintenance system** for industrial milling machines using **machine learning and deep learning techniques**.

The system analyzes real-time sensor data to **predict machine failures before they occur**, enabling proactive maintenance planning, reducing unplanned downtime, and lowering operational costs.  
The project follows a structured pipeline from **data analysis and model training** to **production-style inference and deployment via a Streamlit web application**.

---

## 🎯 Project Objectives
- Predict milling machine failures using sensor and operational data  
- Design and evaluate classical ML and deep learning models  
- Implement a **production-ready ANN inference pipeline**  
- Build an interactive **web-based UI** for real-time and batch predictions  
- Demonstrate an **industry-oriented predictive maintenance solution**

---

## 📊 Dataset
The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains operational data collected from industrial milling machines.

### Features
- **Type** – Machine type (L / M / H)  
- **Air temperature** (K)  
- **Process temperature** (K)  
- **Rotational speed** (rpm)  
- **Torque** (Nm)  
- **Tool wear** (min)  

### Target Variable
- Binary machine failure indicator  
- Failure type classifications (used during analysis and evaluation)

---

## 🧠 Models Implemented

### Phase 2 – Machine Learning
- Random Forest Classifier  
- Data preprocessing and feature engineering  
- Exploratory Data Analysis (EDA)  
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC  

### Phase 3 – Deep Learning (Final Model)
- **Artificial Neural Network (ANN)** built using TensorFlow/Keras  
- Binary classification (Failure / No Failure)  
- Probability-based output with configurable decision threshold  
- Standardized inputs using a trained scaler  
- Exported artifacts for reproducible inference  

---

## ⚙️ System Architecture
**Training → Artifacts → Inference → Web Application**

1. Data preprocessing and feature engineering  
2. ANN model training and evaluation  
3. Export of trained artifacts:
   - ANN model (`.keras`)
   - Feature scaler
   - Feature order
   - Categorical mappings  
4. Python-based inference pipeline  
5. Streamlit web application for end users  

---

## 🖥️ Streamlit Web Application
A fully functional **Streamlit UI** is included for interactive predictive maintenance analysis.

### Key Features
- 🔢 **Single prediction** using manual sensor input  
- 📁 **Batch prediction** via CSV upload  
- 🎚️ Adjustable **failure decision threshold**  
- 📊 Failure probability output  
- ⬇️ Downloadable prediction results  
- Strict CSV schema validation for safe inference  

### Prediction Output
- **Failure probability** (0–100%)  
- **Binary failure decision** based on user-defined threshold  

> ⚠️ This system is intended as a **decision support tool** and should not be used as the sole safety mechanism.

---

## 🚀 Running the Application Locally

### 1️⃣ Install Dependencies
Ensure Python is installed, then run:

```bash
pip install -r requirements.txt
```

### 2️⃣ Launch the Streamlit App
Start the application using:

```bash
streamlit run app.py
```
### 3️⃣ Open in Browser
Navigate to the following URL in your browser:

```bash
http://localhost:8501

```
