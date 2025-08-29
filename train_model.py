import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ---- Load CSV ----
print("Loading and processing dataset...")
df = pd.read_csv("heart.csv")

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Define the target column
target_col = "heartdisease"
if target_col not in df.columns:
    # Try to find the target column if not found
    possible_targets = ["target", "output", "cardio", "tenyearchd", "label", "diagnosis", "class", "heartdisease"]
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        raise ValueError(f"Could not find a target column in {df.columns.tolist()}")

# Define all features we need based on user requirements
print("Identifying features...")

# Categorical features that need encoding
categorical_features = ["sex", "chestpaintype", "restingecg", "exerciseangina", "st_slope"]

# Numerical features that need normalization
numerical_features = ["age", "restingbp", "cholesterol", "maxhr", "oldpeak"]

# Binary features
binary_features = ["fastingbs"]

# Check which features are available in the dataset
available_categorical = [f for f in categorical_features if f in df.columns.str.lower()]
available_numerical = [f for f in numerical_features if f in df.columns.str.lower()]
available_binary = [f for f in binary_features if f in df.columns.str.lower()]

# Combine all available features
available_features = available_categorical + available_numerical + available_binary

print(f"Categorical features: {available_categorical}")
print(f"Numerical features: {available_numerical}")
print(f"Binary features: {available_binary}")

if len(available_features) == 0:
    raise ValueError("No usable features found in CSV.")

# Prepare feature matrix and target vector
X = df[available_features]
y = df[target_col].astype(int)  # ensure binary ints

# Print dataset information
print(f"Dataset shape: {df.shape}")
print(f"Features used: {available_features}")
print(f"Target column: {target_col}")
print(f"Class distribution: \n{df[target_col].value_counts()}")

# Map categorical values if needed (for display purposes later)
feature_mappings = {
    "sex": {"M": 1, "F": 0},
    "chestpaintype": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3},
    "restingecg": {"Normal": 0, "ST": 1, "LVH": 2},
    "exerciseangina": {"N": 0, "Y": 1},
    "st_slope": {"Up": 0, "Flat": 1, "Down": 2}
}

# Apply mappings where needed
for col, mapping in feature_mappings.items():
    if col in X.columns and X[col].dtype == 'object':
        X[col] = X[col].map(mapping)

# ---- Train/test split ----
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Create preprocessing pipeline ----
print("Creating preprocessing pipeline...")

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Define preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, available_numerical),
        ('cat', categorical_transformer, available_categorical)
    ])

# Create the full pipeline with preprocessing and model
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',  # helpful if classes imbalanced
        random_state=42
    ))
])

print("Training model...")
pipe.fit(X_train, y_train)

# ---- Evaluate ----
print("Evaluating model...")
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---- Save trained pipeline ----
print("Saving model...")
with open("model.pkl", "wb") as f:
    pickle.dump({
        "pipeline": pipe,
        "features": available_features,
        "categorical_features": available_categorical,
        "numerical_features": available_numerical,
        "binary_features": available_binary,
        "feature_mappings": feature_mappings
    }, f)

print("Saved model.pkl ✅")
print("\nModel is ready for predictions!")
print("Run app.py to start the web application.")
