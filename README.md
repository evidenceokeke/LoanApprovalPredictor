# Loan Approval Predictor

A supervised ML model that predicts loan approval based on financial features like credit score and loan term. Built with Scikit-learn and Flask, and deployed live on Render with a fully functional web interface.

Try it live: https://loanapprovalpredictor.onrender.com/

**Goal**

Predict whether a loan application will be approved using a lightweight, fast, interpretable model — Logistic Regression.

**Model Details**
* Algorithm: Logistic Regression
* Regularization: L1
* Tuning: GridSearchCV
* Key Feature Insights:
    * cibil_score >= 550 → ~100% approval likelihood
    * loan_term <= 4 years → increased approval chance, even with lower cred

**Dataset**

Source : https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data

Preprocessing, EDA, and feature engineering done in the accompanying Jupyter Notebook.

**Web Interface (Frontend + Backend)**
* Built using Flask, HTML, CSS, and JavaScript
* Real-time prediction via a user-friendly form
* Deployed on Render

**Tech Stack**
* Scikit-learn
* Flask
* HTML/CSS/JavaScript
* Render
* Python

**Run Locally**

```
# Clone the repo
git clone https://github.com/evidenceokeke/LoanApprovalPredictor.git

cd LoanApprovalPredictor

# Install dependencies
pip install -r requirements.txt

# Train the model and save it
python model_training.py

# Run the app
python main.py
```

Visit: http://127.0.0.1:5000
