# Import the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
loan_approval_df = pd.read_csv("../LoanApprovalPredictor2/datasets/loan_approval_dataset.csv")

# There are spaces in the column names and values with strings so strip them
loan_approval_df.columns = loan_approval_df.columns.str.strip()
loan_approval_df['loan_status'] = loan_approval_df['loan_status'].str.strip()
loan_approval_df['education'] = loan_approval_df['education'].str.strip()
loan_approval_df['self_employed'] = loan_approval_df['self_employed'].str.strip()

X = loan_approval_df.drop(columns=['loan_status', 'loan_id'])
y = loan_approval_df['loan_status']

# Encode the features
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Get the numerical features in a list
numeric_features = X.select_dtypes(include="number").columns.tolist()

# Get categorical features in a list
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Define a preprocessing pipeline for the features
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False), categorical_features)
])

# Ensure feature order preservation
preprocessor.set_output(transform="pandas")

# Initialize the logisticRegression model with the best parameters gotten in notebook
logreg_clf = LogisticRegression(random_state=42, C=0.01, penalty="l1", solver="liblinear")

# Define the pipeline
model = Pipeline([
    ('preprocessor', preprocessor),  # Preprocessing step
    ('classifier', logreg_clf)  # Training step
])

model.fit(X, y_encoded)  # Train

# move to pkl file
joblib.dump(
    {
        'model': model,
        'class_names': model.named_steps['classifier'].classes_
    }, "loan_pipeline.pkl"
)
