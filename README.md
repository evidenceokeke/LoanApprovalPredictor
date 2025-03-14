# Loan Approval Predictor

Loan Approval Predictor is a supervised machine learning model built using scikit-learn's 
logistic regression. The model predicts whether a loan application will be approved based on
various features.

**Dataset**

The model uses the Loan-Approval-Prediction-Dataset from Kaggle: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data

**Model Details**
* Algorithm: Logistic Regression
* Hyperparameter Tuning: GridSearchCV
* Regularization: L1 Regularization

**Driving Features**
* Credit Score (cibil_score): Exploratory and feature analysis showed that if you have a credit score
 >= 550, you are almost 100% likely to be approved.
* Loan Term (loan_term): If your loan term is less than or equal to 4 years, you may be approved
  even if your credit score is less than 550.

For more details on data processing, training, and model evaluation, check out the accompanying
Jupyter notebook in the repository.

**Web Interface**

You can interact with the trained model through a web interface deployed on Render. You can
try the model live at: https://loanapprovalpredictor.onrender.com/

**Technologies Used**
* Machine Learning: scikit-learn
* Web Framework: Flask, HTML, CSS, JavaScript
* Deployment: Render



# How to Run Locally

If you'd like to run the model and web interface locally, do this:
1. Clone the repository: git clone https://github.com/evidenceokeke/LoanApprovalPredictor.git
2. Install dependencies: pip install -r requirements.txt
3. Build your model and save on pkl file: python model_training.py
4. Run the Flask app: python main.py (or just do it from your ide)
5. Then click on the link that pops up (http://127.0.0.1:5000)

*Slay! Happy Learning!*
