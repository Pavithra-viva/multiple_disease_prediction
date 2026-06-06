import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'kidney_disease.csv')
if not os.path.exists(csv_path):
    alt_path = os.path.join(base_dir, 'kidney_disease - kidney_disease.csv')
    if os.path.exists(alt_path):
        csv_path = alt_path
    else:
        raise FileNotFoundError(f"Could not find kidney_disease CSV at {csv_path}")

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)
print(df.head())

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop id column if exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Target column
df['classification'] = df['classification'].map({'ckd':1, 'notckd':0})

# Convert numeric-looking string columns to numeric values
numeric_string_cols = ['pcv', 'wc', 'rc']
for col in numeric_string_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Keep actual categorical columns for one-hot encoding
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
if 'classification' in categorical_cols:
    categorical_cols.remove('classification')

if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('classification', axis=1)
y = df['classification']

# Remove features with no observed values, then impute the rest
X = X.loc[:, X.notna().any(axis=0)]
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

model_path = os.path.join(base_dir, 'kidney_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
features_path = os.path.join(base_dir, 'feature_columns.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open(features_path, 'wb') as features_file:
    pickle.dump(X.columns.tolist(), features_file)

print(f"Saved model to: {model_path}")
print(f"Saved scaler to: {scaler_path}")
print(f"Saved feature columns to: {features_path}")