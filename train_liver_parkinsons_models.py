import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

base_dir = os.path.dirname(os.path.abspath(__file__))

# Train Liver Disease model using the app's selected features
liver_csv = os.path.join(base_dir, 'indian_liver_patient - indian_liver_patient.csv')
df_liver = pd.read_csv(liver_csv)

# Map Dataset target to binary labels: 1 = disease, 2 = healthy
if 'Dataset' not in df_liver.columns:
    raise ValueError('Expected Dataset column in liver dataset')
df_liver['target'] = df_liver['Dataset'].map({1: 1, 2: 0})

liver_features = ['Age', 'Total_Bilirubin', 'Alkaline_Phosphotase']
X_liver = df_liver[liver_features].copy()
y_liver = df_liver['target'].copy()

X_liver = X_liver.dropna()
y_liver = y_liver.loc[X_liver.index]

X_train, X_test, y_train, y_test = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)
scaler_liver = StandardScaler().fit(X_train)
X_train_scaled = scaler_liver.transform(X_train)

liver_model = LogisticRegression(max_iter=1000)
liver_model.fit(X_train_scaled, y_train)

X_test_scaled = scaler_liver.transform(X_test)
y_pred = liver_model.predict(X_test_scaled)
print('Liver model accuracy:', accuracy_score(y_test, y_pred))
print('Liver confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Liver classification report:\n', classification_report(y_test, y_pred))

with open(os.path.join(base_dir, 'liver_model.pkl'), 'wb') as f:
    pickle.dump(liver_model, f)
with open(os.path.join(base_dir, 'scaler_liver.pkl'), 'wb') as f:
    pickle.dump(scaler_liver, f)

# Train Parkinsons model using the app's selected features
parkinsons_csv = os.path.join(base_dir, 'parkinsons - parkinsons.csv')
df_parkinsons = pd.read_csv(parkinsons_csv)

parkinsons_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)']
X_parkinsons = df_parkinsons[parkinsons_features].copy()
y_parkinsons = df_parkinsons['status'].copy()

X_parkinsons = X_parkinsons.dropna()
y_parkinsons = y_parkinsons.loc[X_parkinsons.index]

X_train, X_test, y_train, y_test = train_test_split(X_parkinsons, y_parkinsons, test_size=0.2, random_state=42)
scaler_parkinsons = StandardScaler().fit(X_train)
X_train_scaled = scaler_parkinsons.transform(X_train)

parkinsons_model = LogisticRegression(max_iter=1000)
parkinsons_model.fit(X_train_scaled, y_train)

X_test_scaled = scaler_parkinsons.transform(X_test)
y_pred = parkinsons_model.predict(X_test_scaled)
print('Parkinsons model accuracy:', accuracy_score(y_test, y_pred))
print('Parkinsons confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Parkinsons classification report:\n', classification_report(y_test, y_pred))

with open(os.path.join(base_dir, 'parkinsons_model.pkl'), 'wb') as f:
    pickle.dump(parkinsons_model, f)
with open(os.path.join(base_dir, 'scaler_parkinsons.pkl'), 'wb') as f:
    pickle.dump(scaler_parkinsons, f)

print('Saved liver_model.pkl, scaler_liver.pkl, parkinsons_model.pkl, scaler_parkinsons.pkl')
