# 💳 Credit Card Fraud Detection
### *Building a Robust Classifier from Scratch*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Focus-Imbalanced%20Data-orange?style=for-the-badge)

---

## 📌 Project Overview
Detecting fraud in credit card transactions is a "needle in a haystack" problem. In this project, I implemented **Logistic Regression from scratch** to identify fraudulent activity within a highly imbalanced dataset (0.17% fraud rate). 

> **Key takeaway:** Accuracy is misleading here. This project prioritizes **Recall** and **AUPRC** to ensure we actually catch the fraudulent transactions.

---

## 🛠️ The "From Scratch" Engine
Instead of using high-level libraries for the model logic, I coded the mathematical core using **Vectorized NumPy** operations for efficiency.

| Component | Logic |
| :--- | :--- |
| **Activation** | Sigmoid Function $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| **Cost Function** | Log-Loss (Binary Cross-Entropy) |
| **Optimizer** | Gradient Descent |
| **Feature Scaling** | Z-Score Normalization ($\frac{x - \mu}{\sigma}$) |


---
